"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright"):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_returns_empty_when_disabled(tmp_path):
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="auto",
    )
    assert not svc.is_active()
    f = tmp_path / "x.py"
    f.write_text("")
    assert svc.get_diagnostics_sync(str(f)) == []
    svc.shutdown()


def test_service_skips_files_outside_workspace(tmp_path):
    """Files outside any git worktree must not trigger LSP."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    f = tmp_path / "x.py"
    f.write_text("")
    # No .git anywhere — service should report not enabled for this file.
    assert not svc.enabled_for(str(f))
    svc.shutdown()


def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_status_includes_clients(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert info["enabled"] is True
        assert any(c["server_id"] == "pyright" for c in info["clients"])
    finally:
        svc.shutdown()


def test_service_reaps_client_after_idle_timeout(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.2,
    )
    try:
        svc.get_diagnostics_sync(str(f))
        assert svc.get_status()["clients"]
        client = next(iter(svc._clients.values()))
        process = client._proc
        assert process is not None

        deadline = time.monotonic() + 2.0
        while svc.get_status()["clients"] and time.monotonic() < deadline:
            time.sleep(0.02)
        while process.returncode is None and time.monotonic() < deadline:
            time.sleep(0.02)

        assert svc.get_status()["clients"] == []
        assert process.returncode is not None
    finally:
        svc.shutdown()


def test_reused_client_refreshes_last_used_and_survives_reap(mock_pyright):
    """A client re-acquired from the cache must have its ``_last_used``
    timestamp refreshed so a subsequent sweep does NOT evict it.

    Covers the timestamp refresh on the existing-client fast path in
    ``_get_or_spawn`` — without it, a client in constant use would be
    reaped ``idle_timeout`` seconds after its FIRST use.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        first_used = svc._last_used[key]

        # Age the timestamp past the cutoff, then re-acquire the client.
        svc._last_used[key] = first_used - 120.0
        svc.get_diagnostics_sync(str(f))
        assert svc._last_used[key] > first_used - 120.0, (
            "re-acquiring a cached client must refresh _last_used"
        )

        # A sweep right after reuse must keep the client.
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key in svc._clients
        assert svc.get_status()["clients"]
    finally:
        svc.shutdown()


def test_reaper_survives_sweep_error(mock_pyright):
    """One failing sweep must not kill the reaper loop — the loop's
    ``except Exception`` guard must swallow the error and keep sweeping."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    try:
        # Sabotage the sweep itself so the reaper-loop except branch
        # actually runs (a failing client.shutdown() would be swallowed
        # by gather(return_exceptions=True) and never reach the loop).
        calls = {"n": 0}
        real_reap = svc._reap_idle_once

        async def _flaky_reap():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sweep sabotage")
            await real_reap()

        svc._reap_idle_once = _flaky_reap  # type: ignore[method-assign]

        svc.get_diagnostics_sync(str(f))
        assert svc.get_status()["clients"]

        # First sweep raises; later sweeps must still reap the client.
        deadline = time.monotonic() + 3.0
        while svc.get_status()["clients"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert calls["n"] >= 2, "reaper loop died after the failing sweep"
        assert svc.get_status()["clients"] == []
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()
    finally:
        svc.shutdown()


def test_create_from_config_reads_idle_timeout(monkeypatch):
    """``lsp.idle_timeout`` in config.yaml reaches the service."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "idle_timeout": 42}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._idle_timeout == 42.0


def test_create_from_config_invalid_idle_timeout_falls_back(monkeypatch):
    from agent.lsp.manager import DEFAULT_IDLE_TIMEOUT

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "idle_timeout": "not-a-number"}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT


def test_create_from_config_clamps_tiny_idle_timeout(monkeypatch):
    """Sub-floor timeouts are clamped (mid-flight reap could otherwise
    escalate an outer timeout into a permanent broken-set entry); 0 still
    means disabled and is not clamped."""
    from agent.lsp.manager import MIN_IDLE_TIMEOUT

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "idle_timeout": 2}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._idle_timeout == MIN_IDLE_TIMEOUT

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "idle_timeout": 0}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._idle_timeout == 0


def test_default_config_declares_idle_timeout():
    """The canonical default in DEFAULT_CONFIG matches the manager constant
    so config discovery surfaces the knob with the real default value."""
    from agent.lsp.manager import DEFAULT_IDLE_TIMEOUT
    from hermes_cli.config import DEFAULT_CONFIG

    assert float(DEFAULT_CONFIG["lsp"]["idle_timeout"]) == float(DEFAULT_IDLE_TIMEOUT)


# ──────────────────────────────────────────────────────────────────────
# Concurrent-client cap (SCA-4389)
#
# The idle reaper bounds servers by time.  These cover the second bound:
# N simultaneously-active workspace roots must not collectively exceed
# host memory.  Thirteen live tsservers (~1.3 GiB each) on a 16 GiB Mac
# Mini is what took the self-hosted CI runner offline for 5.5h.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pyright_repos(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and yield a factory for git repos.

    Each repo is a distinct workspace root, so each one the service
    touches becomes a separate (server_id, root) cache key — which is
    exactly the per-worktree accumulation this cap has to bound.
    """
    made = []

    def _make(name: str):
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("")
        f = repo / "x.py"
        f.write_text("")
        made.append(repo)
        return f

    first = _make("repo0")
    monkeypatch.chdir(str(first.parent))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield _make
    try:
        next(gen)
    except StopIteration:
        pass


def _svc(**kw):
    base = dict(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,  # cap tests drive eviction, not the clock
    )
    base.update(kw)
    return LSPService(**base)


def test_lru_cap_evicts_least_recently_used(mock_pyright_repos):
    """The (cap + 1)-th root evicts the least-recently-used client.

    This is the criterion that a cache with an eviction path which never
    executes would silently fail — so it asserts on the victim's process
    actually exiting, not merely on dict membership.
    """
    a = mock_pyright_repos("a")
    b = mock_pyright_repos("b")
    c = mock_pyright_repos("c")
    svc = _svc(max_clients=2)
    try:
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(b))
        assert len(svc._clients) == 2

        key_a = next(k for k in svc._clients if str(a.parent) in k[1])
        victim = svc._clients[key_a]
        victim_proc = victim._proc
        assert victim_proc is not None

        # Make A unambiguously the least-recently-used, then spawn C.
        svc._last_used[key_a] = time.time() - 999
        svc.get_diagnostics_sync(str(c))

        assert len(svc._clients) == 2, "cap must hold after the third root"
        assert key_a not in svc._clients, "LRU victim must be evicted"

        deadline = time.monotonic() + 5.0
        while victim_proc.returncode is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert victim_proc.returncode is not None, (
            "evicted client's process must actually exit — an eviction that "
            "only forgets the reference still leaks the ~1.3 GiB server"
        )
    finally:
        svc.shutdown()


def test_lru_cap_zero_disables_eviction(mock_pyright_repos):
    """``max_clients: 0`` means unbounded — the documented escape hatch."""
    a = mock_pyright_repos("a")
    b = mock_pyright_repos("b")
    c = mock_pyright_repos("c")
    svc = _svc(max_clients=0)
    try:
        for f in (a, b, c):
            svc.get_diagnostics_sync(str(f))
        assert len(svc._clients) == 3
    finally:
        svc.shutdown()


def test_cap_eviction_skips_in_flight_client(mock_pyright_repos):
    """Eviction must never shut down a client mid-request.

    A client killed under an in-flight open/wait surfaces as an outer
    timeout, which ``_mark_broken_for_file`` then latches into the
    broken-set for the life of the process — one transient eviction
    would disable LSP for that workspace permanently.
    """
    a = mock_pyright_repos("a")
    b = mock_pyright_repos("b")
    c = mock_pyright_repos("c")
    svc = _svc(max_clients=3)
    try:
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(b))
        assert len(svc._clients) == 2

        key_a = next(k for k in svc._clients if str(a.parent) in k[1])
        key_b = next(k for k in svc._clients if str(b.parent) in k[1])

        # A is the least-recently-used, so it is the natural victim.
        svc._last_used[key_a] = 0.0
        svc._last_used[key_b] = time.time()

        # Tighten the cap so exactly one client must go, with A busy.
        svc._max_clients = 1
        with svc._busy(svc._clients[key_a]):
            svc._loop.run(svc._enforce_max_clients(), timeout=5.0)
            assert key_a in svc._clients, (
                "a busy client must be skipped by cap eviction even when it "
                "is the least-recently-used and the cache is over quota"
            )
            assert key_b not in svc._clients, (
                "eviction must fall through to the next-oldest idle client "
                "rather than skipping the sweep entirely"
            )

        # Skipping a busy client defers eviction, it does not cancel it.
        # Once A drains, the next spawn puts the cache over quota again
        # and A — now idle and least-recently-used — is the victim.
        svc._last_used[key_a] = 0.0
        svc.get_diagnostics_sync(str(c))
        assert key_a not in svc._clients, (
            "after the in-flight request drains the client is evictable"
        )
        assert len(svc._clients) == 1
    finally:
        svc.shutdown()


def test_idle_reaper_skips_in_flight_client(mock_pyright_repos):
    """The idle sweep honours the same in-flight guard as the cap.

    A long ``wait_for_diagnostics`` on a large project can outlast the
    idle cutoff; reaping it mid-wait is the same broken-set trap.
    """
    a = mock_pyright_repos("a")
    svc = _svc(max_clients=0, idle_timeout=60.0)
    try:
        svc.get_diagnostics_sync(str(a))
        key = next(iter(svc._clients))
        client = svc._clients[key]

        svc._last_used[key] = 0.0  # far past the cutoff
        with svc._busy(client):
            svc._loop.run(svc._reap_idle_once(), timeout=5.0)
            assert key in svc._clients, "a busy client must survive the sweep"

        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key not in svc._clients, "an idle, drained client is reaped"
    finally:
        svc.shutdown()


def test_busy_counter_released_on_exception(mock_pyright_repos):
    """``_busy`` must decrement on the error path.

    A leaked count pins the client permanently un-evictable, which
    reintroduces the very leak the cap exists to close.
    """
    a = mock_pyright_repos("a")
    svc = _svc(max_clients=0)
    try:
        svc.get_diagnostics_sync(str(a))
        key = next(iter(svc._clients))
        client = svc._clients[key]

        with pytest.raises(RuntimeError):
            with svc._busy(client):
                assert svc._inflight[key] == 1
                raise RuntimeError("boom")

        assert key not in svc._inflight, "in-flight count must not leak"
        assert not svc._is_busy(key)
    finally:
        svc.shutdown()


def test_get_or_spawn_returns_client_with_reference_held(mock_pyright_repos):
    """``_get_or_spawn`` must hand back a client that is already marked
    in-flight.

    Taking the reference after the lookup lock is released leaves a
    window where a concurrent sweep sees refcount 0 and shuts the client
    down between the lookup and the caller's first request.  Assert the
    reference exists at hand-off on both the spawn and cache-hit paths,
    and that a sweep in that window is a no-op.
    """
    a = mock_pyright_repos("a")
    svc = _svc(max_clients=0, idle_timeout=60.0)
    try:
        # Spawn path.
        client = svc._loop.run(svc._get_or_spawn(str(a)), timeout=10.0)
        assert client is not None
        key = (client.server_id, client.workspace_root)
        assert svc._is_busy(key), "spawn path must return with a reference held"

        # A sweep in the hand-off window must not evict it.
        svc._last_used[key] = 0.0
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key in svc._clients
        svc._release(client)
        assert not svc._is_busy(key)

        # Cache-hit path.
        client2 = svc._loop.run(svc._get_or_spawn(str(a)), timeout=10.0)
        assert client2 is client
        assert svc._is_busy(key), "cache-hit path must return with a reference held"
        svc._release(client2)
        assert not svc._is_busy(key)
    finally:
        svc.shutdown()


def test_default_max_clients_scales_with_host_memory(monkeypatch):
    """The cap is derived from host RAM, not hardcoded — a 16 GiB Mac
    Mini and a 128 GiB workstation must not get the same number."""
    from agent.lsp import manager as mgr

    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 16.0)
    assert mgr.default_max_clients() == 3

    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 128.0)
    assert mgr.default_max_clients() == 24

    # Unknown host memory falls back to the floor rather than guessing.
    monkeypatch.setattr(mgr, "_host_total_gib", lambda: None)
    assert mgr.default_max_clients() == mgr.MIN_MAX_CLIENTS

    # Tiny hosts still get a workable floor, never 0.
    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 2.0)
    assert mgr.default_max_clients() == mgr.MIN_MAX_CLIENTS

    # Huge hosts are bounded by the sanity ceiling.
    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 4096.0)
    assert mgr.default_max_clients() == mgr.MAX_MAX_CLIENTS


def test_create_from_config_reads_max_clients(monkeypatch):
    """``lsp.max_clients`` in config.yaml reaches the service."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "max_clients": 7}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_clients == 7


def test_create_from_config_max_clients_absent_derives_from_host(monkeypatch):
    """Absent config → derived default, not a hardcoded constant."""
    from agent.lsp import manager as mgr

    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 16.0)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    assert svc._max_clients == 3


def test_create_from_config_clamps_and_rejects_bad_max_clients(monkeypatch):
    """A cap of 1 thrashes (alternating edits evict each other); 0 still
    disables; garbage falls back to the derived default."""
    from agent.lsp import manager as mgr

    monkeypatch.setattr(mgr, "_host_total_gib", lambda: 16.0)

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "max_clients": 1}},
    )
    assert LSPService.create_from_config()._max_clients == mgr.MIN_MAX_CLIENTS

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "max_clients": 0}},
    )
    assert LSPService.create_from_config()._max_clients == 0

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "max_clients": "not-a-number"}},
    )
    assert LSPService.create_from_config()._max_clients == 3


def test_default_config_declares_max_clients():
    """The knob is discoverable via DEFAULT_CONFIG, declared as null so
    the host-derived default applies unless the user overrides it."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert "max_clients" in DEFAULT_CONFIG["lsp"]
    assert DEFAULT_CONFIG["lsp"]["max_clients"] is None


def test_status_reports_cap_and_timeout(mock_pyright_repos):
    """``hermes lsp status`` must surface both bounds — otherwise "the
    cache never evicts" stays unfalsifiable from outside the process."""
    a = mock_pyright_repos("a")
    svc = _svc(max_clients=5, idle_timeout=123.0)
    try:
        svc.get_diagnostics_sync(str(a))
        status = svc.get_status()
        assert status["max_clients"] == 5
        assert status["idle_timeout"] == 123.0
    finally:
        svc.shutdown()


