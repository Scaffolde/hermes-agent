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
import os
import threading
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
        # Monotonic, matching the service: eviction ages must not be read
        # off the wall clock, or an NTP step re-creates unbounded retention.
        svc._last_used[key] = svc._now() - age
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
# Config plumbing, through the REAL loader
#
# Every test above stubs ``load_config`` and hands ``create_from_config``
# the already-resolved dict it expects.  That proves the manager reads
# the dict correctly and nothing more: it cannot catch a key that never
# survives ``DEFAULT_CONFIG`` merging, a ``HERMES_HOME`` that resolves
# somewhere else, or the (path, mtime_ns, size) cache in ``load_config``
# handing back a stale entry.  A knob can be perfectly parsed here and
# still be unreachable from a real ``config.yaml`` on disk.
#
# These two write an actual file under a temporary ``HERMES_HOME`` and
# stub nothing, so the assertion covers the whole path a user's edit
# actually takes.  (Raised by Codex review on the sibling PR #51, where
# it applies to this branch equally.)
# ──────────────────────────────────────────────────────────────────────


def _write_hermes_config(tmp_path, monkeypatch, body: str) -> None:
    """Point HERMES_HOME at *tmp_path* and write a real config.yaml there."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # HERMES_HOME_MODE would otherwise be free to redirect the home out
    # from under the test on a host that sets it.
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


def test_eviction_bounds_survive_the_real_config_loader(tmp_path, monkeypatch):
    """An ``lsp:`` block written to a real config.yaml reaches the service.

    Non-default values on all three knobs, so a default leaking through
    (or the block being dropped in merging) fails rather than
    coincidentally matching.
    """
    _write_hermes_config(
        tmp_path,
        monkeypatch,
        "lsp:\n"
        "  enabled: false\n"
        "  idle_timeout: 37\n"
        "  sweep_interval: 11\n"
        "  max_clients: 5\n",
    )

    svc = LSPService.create_from_config()
    assert svc is not None, "create_from_config() returned None for a real config.yaml"
    try:
        assert svc._idle_timeout == 37.0
        assert svc._sweep_interval == 11.0
        assert svc._max_clients == 5
    finally:
        svc.shutdown()


def test_real_config_without_an_lsp_block_still_gets_the_bounds(tmp_path, monkeypatch):
    """Omitting ``lsp:`` entirely must fall back to the derived defaults.

    This is the case that actually ships: nobody writes the block by
    hand.  If defaults only materialised via a stubbed dict, the bound
    would be absent exactly where the leak happened.
    """
    _write_hermes_config(tmp_path, monkeypatch, "model:\n  default: test\n")

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._sweep_interval == DEFAULT_SWEEP_INTERVAL
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


# ---------------------------------------------------------------------------
# Review-round fixes (Codex review on PR #50).
#
# Each test below fails against the code as first pushed.  They are grouped
# here because they share one theme: the *first* implementation bounded the
# population only at spawn time and read its ages off the wall clock, so
# several ordinary conditions restored the unbounded retention the feature
# exists to prevent.
# ---------------------------------------------------------------------------


def test_cap_is_re_enforced_after_a_busy_burst_drains():
    """Overage created while every client was in-flight must not survive.

    The spawn-time enforcement finds no evictable victim when a burst
    touches more roots than the cap allows, so the overage outlives the
    burst.  Sweeping cannot collect it either: the clients were just
    touched.  The reaper has to re-check the cap, not only idleness.
    """
    svc = _service(idle_timeout=600.0, max_clients=2, sweep_interval=0.05)
    try:
        keys = [_inject(svc, f"/tmp/burst-{i}", age=0.0)[0] for i in range(5)]
        # Every root is mid-request: nothing is evictable right now.
        for k in keys:
            svc._acquire(k)
        assert svc.enforce_cap_now() == []
        assert len(svc._clients) == 5

        # The burst drains.  Nothing is idle (all just used), so only cap
        # enforcement can reclaim them.
        for k in keys:
            svc._release(k)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(svc._clients) > 2:
            time.sleep(0.05)
        assert len(svc._clients) == 2, (
            "reaper never re-enforced the cap after the burst drained"
        )
    finally:
        svc.shutdown()


def test_cap_is_enforced_even_when_the_idle_timeout_is_disabled():
    """``idle_timeout: 0`` opts out of idleness, never out of the cap.

    With no reaper running at all (the original gate started it only when
    idle_timeout > 0), an over-cap population left by a burst was permanent.
    """
    svc = _service(idle_timeout=0.0, max_clients=2, sweep_interval=0.05)
    try:
        assert svc._reaper is not None, "no reaper runs, so the cap has no enforcer"
        for i in range(5):
            _inject(svc, f"/tmp/nosweep-{i}", age=0.0)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(svc._clients) > 2:
            time.sleep(0.05)
        assert len(svc._clients) == 2
    finally:
        svc.shutdown()


def test_eviction_ages_survive_a_wall_clock_jump(monkeypatch):
    """An NTP step backwards must not suspend eviction.

    Seeded through ``svc._now()`` so the test cannot silently compare two
    different clocks.  Against a wall-clock implementation the backward
    step leaves the stored stamp ahead of the cutoff and nothing is
    evicted until real time catches up — exactly the unbounded retention
    this feature exists to prevent.
    """
    svc = _service(idle_timeout=10.0, max_clients=99, sweep_interval=3600.0)
    try:
        key, client = _inject(svc, "/tmp/ntp", age=60.0)

        # The wall clock jumps an hour backwards; monotonic cannot.
        real_time = time.time
        monkeypatch.setattr(time, "time", lambda: real_time() - 3600.0)

        assert svc.sweep_idle_now() == [key], (
            "a backward wall-clock step suspended eviction"
        )
        assert client.shutdown_calls == 1
    finally:
        svc.shutdown()


def test_non_finite_bounds_fall_back_instead_of_crashing_the_lsp_path(monkeypatch):
    """``max_clients: .nan`` is valid YAML and must not break write/patch.

    ``float('.nan')`` survives the parse and slips past ``<= 0``; the
    ``int()`` at the call site then raises, and because ``get_service()``
    does not catch factory exceptions the whole LSP path dies.
    """
    for bad in (".nan", ".inf", float("nan"), float("inf")):
        cfg = {"lsp": {"enabled": False, "max_clients": bad, "idle_timeout": bad}}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda cfg=cfg: cfg)
        svc = LSPService.create_from_config()
        assert svc is not None, f"{bad!r} took down create_from_config()"
        try:
            assert svc._max_clients == default_max_clients()
            assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        finally:
            svc.shutdown()


def test_cap_derivation_respects_a_cgroup_limit(monkeypatch, tmp_path):
    """A 4 GiB container on a big node must not derive the node's cap."""
    from agent.lsp import manager as mgr

    node_bytes = 64 * 1024 ** 3
    container_bytes = 4 * 1024 ** 3
    monkeypatch.setattr(mgr, "_cgroup_memory_limit_bytes", lambda: container_bytes)
    monkeypatch.setattr(os, "sysconf", lambda name: {
        "SC_PHYS_PAGES": node_bytes // 4096, "SC_PAGE_SIZE": 4096,
    }[name])

    assert mgr.host_memory_bytes() == container_bytes
    # The node-derived cap would permit far more than the container holds.
    assert default_max_clients() < default_max_clients(node_bytes)


def test_cgroup_unlimited_sentinels_are_ignored(tmp_path):
    """cgroup v2 ``max`` and v1's LONG_MAX sentinel mean 'no limit'.

    Treating either as a real ceiling would derive a cap from a nonsense
    number instead of falling through to the sysconf total.
    """
    from agent.lsp import manager as mgr

    for sentinel in ("max", str(2 ** 63 - 1), "", "0", "not-a-number"):
        p = tmp_path / f"limit-{abs(hash(sentinel))}"
        p.write_text(sentinel)
        assert mgr._cgroup_memory_limit_bytes((str(p),)) is None, sentinel

    # A real limit is honoured.
    real = tmp_path / "real"
    real.write_text(str(4 * 1024 ** 3))
    assert mgr._cgroup_memory_limit_bytes((str(real),)) == 4 * 1024 ** 3

    # A missing path is simply absent, not an error.
    assert mgr._cgroup_memory_limit_bytes((str(tmp_path / "nope"),)) is None


def test_spawn_does_not_wait_on_the_evicted_victims_shutdown():
    """Cap eviction must not be charged to the request that just spawned.

    A slow victim shutdown awaited inline pushes the caller past its outer
    diagnostic budget; the wrapper then times out and marks the brand-new
    root broken for the life of the service.
    """
    import asyncio as _asyncio

    svc = _service(idle_timeout=600.0, max_clients=1, sweep_interval=3600.0)
    try:
        slow_key, slow_client = _inject(svc, "/tmp/slow-victim", age=100.0)

        released = threading.Event()

        async def _slow_shutdown():
            await _asyncio.sleep(2.0)
            slow_client.shutdown_calls += 1
            released.set()

        slow_client.shutdown = _slow_shutdown

        started = time.monotonic()
        # Simulate the tail of _get_or_spawn: a new client lands, then cap
        # enforcement is kicked off in the background rather than awaited.
        new_key, _ = _inject(svc, "/tmp/fresh", age=0.0)
        svc._loop.spawn(svc._enforce_cap_async(protect=new_key))
        elapsed = time.monotonic() - started

        assert elapsed < 1.0, (
            f"spawn path blocked {elapsed:.2f}s on the victim's shutdown"
        )
        assert released.wait(timeout=6.0), "background eviction never ran"
    finally:
        svc.shutdown()


def test_human_readable_status_renders_the_bounds(capsys):
    """``hermes lsp status`` (no --json) must show the bounds and ages.

    The operator auditability this feature claims is worthless if only the
    machine-readable path carries it.
    """
    from agent.lsp import cli as lsp_cli

    svc = _service(idle_timeout=123.0, max_clients=7, sweep_interval=45.0)
    try:
        _inject(svc, "/tmp/ws-render", age=5.0)
        info = svc.get_status()
    finally:
        svc.shutdown()

    rendered = "\n".join([
        f"  idle_timeout:    {info['idle_timeout']}s",
        f"  sweep_interval:  {info['sweep_interval']}s",
        f"  max_clients:     {info['max_clients']}",
    ])
    # Guard the contract the CLI formats against, so a rename of these
    # status keys fails here rather than silently blanking the display.
    assert "123.0" in rendered and "7" in rendered and "45.0" in rendered
    assert info["clients"][0]["idle_seconds"] >= 5.0
    assert info["clients"][0]["inflight"] == 0
    assert hasattr(lsp_cli, "_cmd_status")


# ---------------------------------------------------------------------------
# round 2 — defects found verifying the Codex review threads (SCA-4389)
# ---------------------------------------------------------------------------


def test_reaper_interval_honours_sweep_interval_when_idle_is_disabled():
    """``idle_timeout: 0`` must not collapse the reaper into a 1s busy-poll.

    The interval is clamped by the idle timeout so a server cannot sit idle
    well past its deadline waiting for the next sweep.  That clamp is
    meaningless once idleness is opted out: only the cap still needs
    re-checking, and the cap has no deadline.  Feeding 0 through the clamp
    drove the interval to the 1.0s floor, so the documented "keep every
    server warm" setting silently bought a wakeup every second for the life
    of the process.
    """
    pinned = _service(idle_timeout=0.0, sweep_interval=60.0)
    try:
        assert pinned._reaper_interval() == 60.0
    finally:
        pinned.shutdown()

    # The clamp must still bite when idleness *is* enabled: a 5s timeout
    # cannot wait 60s for its sweep.
    tight = _service(idle_timeout=5.0, sweep_interval=60.0)
    try:
        assert tight._reaper_interval() == 5.0
    finally:
        tight.shutdown()

    # ...and the sweep interval wins when it is the smaller of the two.
    normal = _service(idle_timeout=600.0, sweep_interval=60.0)
    try:
        assert normal._reaper_interval() == 60.0
    finally:
        normal.shutdown()


def test_config_max_clients_zero_is_rejected_loudly(monkeypatch, caplog):
    """A non-positive cap falls back to the host default — and says so.

    ``max_clients: 0`` reads like "no limit" and the user guide said exactly
    that, but the cap is deliberately not disableable: this whole feature
    exists because an unbounded population put the host into swap and took
    self-hosted CI offline.  Falling back is the right behaviour; doing it
    *silently* is not, because the operator then believes a bound they asked
    to remove is gone when it is still enforced (and vice versa).
    """
    cfg = {"lsp": {"enabled": False, "max_clients": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    with caplog.at_level(logging.WARNING, logger="agent.lsp.manager"):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._max_clients == default_max_clients()
    finally:
        svc.shutdown()

    assert any(
        "max_clients" in r.getMessage() for r in caplog.records
    ), "a rejected cap must be logged, not swallowed"


def test_user_guide_does_not_promise_a_disableable_cap():
    """The docs must not document a knob the code deliberately refuses.

    ``max_clients: 0`` was documented as the way to disable the cap; the
    coercion treats 0 as garbage and substitutes the host default.  A doc
    that contradicts the code is how an operator "disables" a bound and
    never learns it is still on.
    """
    from pathlib import Path

    doc = Path(__file__).resolve().parents[3] / (
        "website/docs/user-guide/features/lsp.md"
    )
    text = doc.read_text(encoding="utf-8")
    assert "`max_clients: 0` to disable the cap" not in text
    assert "max_clients" in text, "guard the path, not just the absence"
