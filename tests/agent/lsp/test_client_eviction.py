"""Tests for LSP client eviction — the idle reaper and the LRU cap.

Background: ``LSPService`` kept one language server per
``(server_id, workspace_root)`` for the life of the process.  It
recorded ``_last_used`` timestamps and carried a
``DEFAULT_IDLE_TIMEOUT = 600`` constant whose comment claimed
"servers idle for >10min get reaped" — but nothing ever read either
one.  On a multi-worktree host that meant an unbounded ratchet: 13
live ``typescript-language-server`` processes holding ~16 GiB on a
16 GiB Mac Mini, which is what pushed the box into swap and took the
self-hosted CI runner offline.

The bookkeeping existing while the eviction never ran is the same
false-green shape as a ``--dry-run`` guard that skips the check it
claims to make.  So these tests deliberately do NOT settle for calling
the sweep by hand: :func:`test_background_reaper_evicts_without_manual_sweep`
constructs a service and waits, proving the reaper is actually wired
to a timer.  A hand-called sweep would pass just as happily against
the broken code plus one unreferenced method.

Covered:
- host-memory-derived cap defaults (and their clamps)
- idle eviction fires, via the background reaper and via the sync helper
- the LRU cap evicts the least-recently-used root, not an arbitrary one
- in-flight requests are never evicted mid-flight
- eviction is observable in the event log, with a reason
- a re-spawn after eviction re-announces (the INFO dedup is cleared)
"""
from __future__ import annotations

import logging
import time

import pytest

from agent.lsp import eventlog
from agent.lsp.manager import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_SWEEP_INTERVAL,
    LSP_CLIENT_FOOTPRINT_BYTES,
    LSP_MEMORY_BUDGET_FRACTION,
    MAX_CLIENT_CAP,
    MIN_CLIENT_CAP,
    LSPService,
    default_max_clients,
)


class _FakeClient:
    """Stands in for :class:`agent.lsp.client.LSPClient`.

    Only the surface the manager touches during eviction: identity,
    liveness, and an awaitable ``shutdown`` that records that it ran.
    """

    def __init__(self, server_id: str, workspace_root: str) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.shutdown_calls = 0
        self._running = True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> str:
        return "running" if self._running else "stopped"

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._running = False


def _service(**kw) -> LSPService:
    """An enabled service with the spawn path unused — we inject clients."""
    kw.setdefault("enabled", True)
    kw.setdefault("wait_mode", "document")
    kw.setdefault("wait_timeout", 2.0)
    kw.setdefault("install_strategy", "manual")
    return LSPService(**kw)


def _inject(svc: LSPService, root: str, *, age: float = 0.0, server_id: str = "typescript"):
    """Register a fake client as if it had been spawned ``age`` seconds ago."""
    client = _FakeClient(server_id, root)
    key = (server_id, root)
    with svc._state_lock:
        svc._clients[key] = client
        svc._last_used[key] = time.time() - age
    return key, client


# ---------------------------------------------------------------------------
# memory-derived defaults
# ---------------------------------------------------------------------------


def test_default_cap_scales_with_host_memory():
    """The cap is derived from host RAM, not hardcoded — a 16 GiB Mac Mini
    and a 128 GiB workstation must not get the same number."""
    small = default_max_clients(total_bytes=16 * 1024**3)
    large = default_max_clients(total_bytes=128 * 1024**3)
    assert small < large, "cap must scale with host memory"

    # 16 GiB * 0.25 budget / ~1.3 GiB per server ≈ 3.
    expected_small = int((16 * 1024**3) * LSP_MEMORY_BUDGET_FRACTION // LSP_CLIENT_FOOTPRINT_BYTES)
    assert small == expected_small

    # The measured incident had 13 servers live on a 16 GiB host.  Whatever
    # the arithmetic, the derived cap for that host must be well under it.
    assert small < 13


def test_default_cap_is_clamped_both_ends():
    """A tiny host still gets one server; a huge host doesn't get an
    unbounded pool just because it has RAM to burn."""
    assert default_max_clients(total_bytes=1024**3) == MIN_CLIENT_CAP
    assert default_max_clients(total_bytes=4096 * 1024**3) == MAX_CLIENT_CAP


def test_default_cap_falls_back_when_memory_unknown():
    """Undetectable host memory must not crash or yield an unbounded cap."""
    cap = default_max_clients(total_bytes=None)
    assert MIN_CLIENT_CAP <= cap <= MAX_CLIENT_CAP


# ---------------------------------------------------------------------------
# idle eviction
# ---------------------------------------------------------------------------


def test_background_reaper_evicts_without_manual_sweep():
    """THE regression test: eviction must fire off the service's own timer.

    Nothing in this test calls the sweep.  It constructs a service with a
    fast sweep interval, injects a client that is already past its idle
    timeout, and waits.  Against the pre-fix code — which had
    ``_last_used``, ``_idle_timeout`` and a reaping *comment* but no
    reaper — this fails.
    """
    svc = _service(idle_timeout=0.2, sweep_interval=0.05)
    try:
        key, client = _inject(svc, "/tmp/ws-idle", age=5.0)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            with svc._state_lock:
                gone = key not in svc._clients
            if gone:
                break
            time.sleep(0.05)

        with svc._state_lock:
            assert key not in svc._clients, "idle client was never reaped by the timer"
            assert key not in svc._last_used, "eviction left stale bookkeeping behind"
        assert client.shutdown_calls == 1, "evicted client's process was not shut down"
    finally:
        svc.shutdown()


def test_sweep_leaves_recently_used_clients_alone():
    """Eviction is idleness-based, not indiscriminate."""
    svc = _service(idle_timeout=600.0, sweep_interval=3600.0)
    try:
        fresh_key, fresh = _inject(svc, "/tmp/ws-fresh", age=1.0)
        stale_key, stale = _inject(svc, "/tmp/ws-stale", age=900.0)

        evicted = svc.sweep_idle_now()

        assert stale_key in evicted
        assert fresh_key not in evicted
        with svc._state_lock:
            assert fresh_key in svc._clients
            assert stale_key not in svc._clients
        assert fresh.shutdown_calls == 0
        assert stale.shutdown_calls == 1
    finally:
        svc.shutdown()


def test_idle_timeout_zero_disables_reaping():
    """An explicit 0 opts out — some hosts genuinely want servers pinned."""
    svc = _service(idle_timeout=0.0, sweep_interval=3600.0)
    try:
        key, client = _inject(svc, "/tmp/ws-pinned", age=99999.0)
        assert svc.sweep_idle_now() == []
        with svc._state_lock:
            assert key in svc._clients
        assert client.shutdown_calls == 0
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# LRU cap
# ---------------------------------------------------------------------------


def test_cap_evicts_least_recently_used_root():
    """The (cap + 1)-th root evicts the LRU one — bounding the population
    independently of whether anything is idle."""
    svc = _service(idle_timeout=600.0, max_clients=2, sweep_interval=3600.0)
    try:
        oldest_key, oldest = _inject(svc, "/tmp/ws-a", age=300.0)
        middle_key, middle = _inject(svc, "/tmp/ws-b", age=200.0)
        newest_key, newest = _inject(svc, "/tmp/ws-c", age=1.0)

        # Three clients, cap of two: exactly one eviction, and it must be
        # the least-recently-used root rather than an arbitrary victim.
        evicted = svc.enforce_cap_now()

        assert evicted == [oldest_key]
        with svc._state_lock:
            assert oldest_key not in svc._clients
            assert middle_key in svc._clients
            assert newest_key in svc._clients
        assert oldest.shutdown_calls == 1
        assert middle.shutdown_calls == 0
        assert newest.shutdown_calls == 0
    finally:
        svc.shutdown()


def test_cap_evicts_down_to_the_limit_not_just_once():
    """Overshooting the cap by several roots drains to the cap, not to
    cap+n-1."""
    svc = _service(idle_timeout=600.0, max_clients=2, sweep_interval=3600.0)
    try:
        for i in range(5):
            _inject(svc, f"/tmp/ws-{i}", age=100.0 - i)
        svc.enforce_cap_now()
        with svc._state_lock:
            assert len(svc._clients) == 2
    finally:
        svc.shutdown()


def test_cap_under_limit_is_a_no_op():
    svc = _service(idle_timeout=600.0, max_clients=4, sweep_interval=3600.0)
    try:
        _inject(svc, "/tmp/ws-a", age=300.0)
        _inject(svc, "/tmp/ws-b", age=200.0)
        assert svc.enforce_cap_now() == []
        with svc._state_lock:
            assert len(svc._clients) == 2
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# in-flight safety
# ---------------------------------------------------------------------------


def test_idle_sweep_never_evicts_an_inflight_client():
    """Criterion 6: in-flight work drains first.  A client with an
    outstanding request is not killed mid-request even when idle."""
    svc = _service(idle_timeout=0.1, sweep_interval=3600.0)
    try:
        key, client = _inject(svc, "/tmp/ws-busy", age=900.0)
        svc._acquire(key)

        assert svc.sweep_idle_now() == []
        with svc._state_lock:
            assert key in svc._clients
        assert client.shutdown_calls == 0

        # Once the request drains, the next sweep collects it.
        svc._release(key)
        assert key in svc.sweep_idle_now()
        assert client.shutdown_calls == 1
    finally:
        svc.shutdown()


def test_cap_never_evicts_an_inflight_client():
    """The cap must also respect in-flight work: it evicts the LRU
    *evictable* root, skipping busy ones."""
    svc = _service(idle_timeout=600.0, max_clients=1, sweep_interval=3600.0)
    try:
        busy_key, busy = _inject(svc, "/tmp/ws-busy", age=900.0)
        idle_key, idle = _inject(svc, "/tmp/ws-idle", age=10.0)
        svc._acquire(busy_key)

        evicted = svc.enforce_cap_now()

        assert evicted == [idle_key], "cap evicted the in-flight client"
        assert busy.shutdown_calls == 0
        assert idle.shutdown_calls == 1
    finally:
        svc.shutdown()


def test_acquire_release_refcount_is_balanced():
    """Nested/concurrent requests against one root must not let the first
    release expose the client to eviction while the second is live."""
    svc = _service(idle_timeout=0.1, sweep_interval=3600.0)
    try:
        key, client = _inject(svc, "/tmp/ws-nested", age=900.0)
        svc._acquire(key)
        svc._acquire(key)
        svc._release(key)

        assert svc.sweep_idle_now() == [], "evicted while a request was still live"

        svc._release(key)
        assert key in svc.sweep_idle_now()
        assert client.shutdown_calls == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def test_eviction_logs_the_root_and_the_reason(caplog):
    """Criterion 4: 'the cache never evicts' must be falsifiable from the
    log rather than reconstructed from ``ps``."""
    svc = _service(idle_timeout=0.1, sweep_interval=3600.0)
    try:
        _inject(svc, "/tmp/ws-observable", age=900.0)
        with caplog.at_level(logging.INFO, logger="hermes.lint.lsp"):
            svc.sweep_idle_now()

        messages = [r.getMessage() for r in caplog.records]
        assert any("/tmp/ws-observable" in m for m in messages), messages
        assert any("evicted" in m and "idle" in m for m in messages), messages
    finally:
        svc.shutdown()


def test_cap_eviction_names_its_distinct_reason(caplog):
    """An LRU eviction and an idle eviction must be distinguishable in the
    log — they mean different things about how the host is loaded."""
    svc = _service(idle_timeout=600.0, max_clients=1, sweep_interval=3600.0)
    try:
        _inject(svc, "/tmp/ws-lru-victim", age=900.0)
        _inject(svc, "/tmp/ws-keeper", age=1.0)
        with caplog.at_level(logging.INFO, logger="hermes.lint.lsp"):
            svc.enforce_cap_now()

        messages = [r.getMessage() for r in caplog.records]
        assert any("/tmp/ws-lru-victim" in m and "cap" in m for m in messages), messages
    finally:
        svc.shutdown()


def test_respawn_after_eviction_reannounces_active(caplog):
    """``log_active`` dedups INFO per (server_id, root) forever.  Without
    clearing that on eviction, a root that is evicted and legitimately
    re-spawned would silently drop to DEBUG and the log would imply the
    server had been up the whole time."""
    eventlog.reset_announce_caches()
    svc = _service(idle_timeout=0.1, sweep_interval=3600.0)
    try:
        root = "/tmp/ws-respawn"
        with caplog.at_level(logging.INFO, logger="hermes.lint.lsp"):
            eventlog.log_active("typescript", root)
            _inject(svc, root, age=900.0)
            svc.sweep_idle_now()
            caplog.clear()
            eventlog.log_active("typescript", root)

        messages = [r.getMessage() for r in caplog.records]
        assert any("active for" in m for m in messages), messages
    finally:
        svc.shutdown()
        eventlog.reset_announce_caches()


def test_status_reports_the_bounds(tmp_path):
    """``hermes lsp status`` should show the limits that are actually in
    force, so an operator can see the cap without reading source."""
    svc = _service(idle_timeout=123.0, max_clients=7, sweep_interval=3600.0)
    try:
        _inject(svc, "/tmp/ws-status", age=5.0)
        status = svc.get_status()
        assert status["idle_timeout"] == 123.0
        assert status["max_clients"] == 7
        assert status["clients"][0]["idle_seconds"] >= 5.0
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------


def test_create_from_config_honours_lsp_bounds(monkeypatch):
    """Criterion 3: both bounds are configurable.  Before this change
    ``create_from_config`` never passed an idle timeout at all, so the
    constant was unreachable from config."""
    cfg = {"lsp": {"enabled": False, "idle_timeout": 42, "max_clients": 3}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == 42.0
        assert svc._max_clients == 3
    finally:
        svc.shutdown()


def test_create_from_config_defaults_are_memory_derived(monkeypatch):
    cfg = {"lsp": {"enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._max_clients == default_max_clients()
    finally:
        svc.shutdown()


@pytest.mark.parametrize("bad", ["nonsense", -5, None, {}])
def test_create_from_config_rejects_bad_bounds(monkeypatch, bad):
    """Garbage in config must fall back to the safe default rather than
    disabling the bound (which is how you silently get the old behaviour
    back)."""
    cfg = {"lsp": {"enabled": False, "max_clients": bad}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert MIN_CLIENT_CAP <= svc._max_clients <= MAX_CLIENT_CAP
    finally:
        svc.shutdown()


# ──────────────────────────────────────────────────────────────────────
# Config discoverability
#
# The bounds are readable from config.yaml, but a knob that isn't in
# DEFAULT_CONFIG is undiscoverable: `hermes config` won't list it and
# users can't find it without reading the source.  These pin the
# declaration to the manager's own constants so the two can't drift.
# ──────────────────────────────────────────────────────────────────────


def test_default_config_declares_eviction_bounds():
    from hermes_cli.config import DEFAULT_CONFIG

    lsp = DEFAULT_CONFIG["lsp"]
    for key in ("idle_timeout", "sweep_interval", "max_clients"):
        assert key in lsp, f"lsp.{key} must be declared in DEFAULT_CONFIG"


def test_default_config_bounds_match_manager_constants():
    """A declared default that disagrees with the code is worse than no
    default — it documents behaviour the service doesn't have."""
    from hermes_cli.config import DEFAULT_CONFIG

    lsp = DEFAULT_CONFIG["lsp"]
    assert float(lsp["idle_timeout"]) == float(DEFAULT_IDLE_TIMEOUT)
    assert float(lsp["sweep_interval"]) == float(DEFAULT_SWEEP_INTERVAL)
    # null means "measure this host", which is what create_from_config
    # translates into default_max_clients().
    assert lsp["max_clients"] is None


def test_declared_default_config_round_trips_through_the_service(monkeypatch):
    """Feeding DEFAULT_CONFIG back in must reproduce the shipped defaults.

    Catches a declaration that parses but lands on different values than
    an absent config would.
    """
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = {"lsp": dict(DEFAULT_CONFIG["lsp"])}
    cfg["lsp"]["enabled"] = False
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._max_clients == default_max_clients()
    finally:
        svc.shutdown()
