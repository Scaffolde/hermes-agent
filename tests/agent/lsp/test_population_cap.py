"""The LSP client population is bounded by count, not only by idleness.

SCA-4389.  ``main`` already reaps *idle* clients (``_reap_idle_once``),
which bounds a fleet nobody is touching.  It does not bound a fleet
everybody is touching: N simultaneously-active worktrees hold N
language servers, and one ``typescript-language-server`` against a
large TypeScript checkout costs ~1.3 GiB.  Thirteen live servers held
~16 GiB on a 16 GiB host, pushed the box into swap, pushed free disk
under the CI runner's admission floor, and took self-hosted CI offline
for 5.5h.  Every one of those thirteen had a fresh ``_last_used``, so
the idle reaper was working as designed and still could not help.

These tests are deliberately not "the policy function returns the right
list".  A cache with an eviction path that never executes is exactly
the false green this defect was, so the headline tests drive real
spawned mock servers through the real request path and assert the OS
process is gone — each with a positive control that runs the identical
scenario with the bound disabled.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from agent.lsp.manager import (
    LSPService,
    default_max_servers,
    MAX_DERIVED_MAX_SERVERS,
    MIN_DERIVED_MAX_SERVERS,
)
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")

GIB = 1024 ** 3


# ----------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------


@pytest.fixture
def mock_roots(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and yield a factory for git roots.

    Each root is a separate fake repo, so each maps to a distinct
    ``(server_id, workspace_root)`` key and therefore its own client.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": "errors"},
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock pyright",
    )

    made: list[Path] = []

    def make_root(name: str) -> Path:
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("")
        f = repo / "x.py"
        f.write_text("print('hi')\n")
        made.append(repo)
        return f

    try:
        yield make_root
    finally:
        SERVERS[target_index] = original


def _pids(svc: LSPService) -> dict:
    """Live OS pids per client key, for proving a process really died."""
    out = {}
    for key, client in list(svc._clients.items()):
        proc = getattr(client, "_proc", None)
        if proc is not None and proc.pid:
            out[key] = proc.pid
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


# ----------------------------------------------------------------------
# the derived default
# ----------------------------------------------------------------------


def test_default_cap_is_derived_from_host_memory(monkeypatch):
    """A 16 GiB Mac Mini and a 128 GiB workstation must not get the
    same number — that is the entire point of deriving rather than
    hardcoding, and hardcoding high is what produced this defect."""
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: 16 * GIB)
    small = default_max_servers()
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: 128 * GIB)
    large = default_max_servers()

    assert small < large
    # 16 GiB * 0.25 / 1.3 GB -> 3.  The measured outage held 13 servers
    # on exactly this host, so the derived cap must be well under it.
    assert small == 3
    assert small < 13, "a 16 GiB host must not permit the fleet that broke it"
    assert large == 26
    assert MIN_DERIVED_MAX_SERVERS <= small <= MAX_DERIVED_MAX_SERVERS
    assert MIN_DERIVED_MAX_SERVERS <= large <= MAX_DERIVED_MAX_SERVERS


def test_undiscoverable_host_memory_assumes_small(monkeypatch):
    """Guessing high is what produced the defect, so an unmeasurable
    host gets a conservative fixed cap, never an optimistic one."""
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: None)
    assert default_max_servers() == 4
    # Conservative means "well under what a large host would allow" —
    # the fallback must not silently out-permit a measured big machine.
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: 128 * GIB)
    assert default_max_servers() > 4


def test_derived_cap_is_clamped_at_both_ends(monkeypatch):
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: 1 * GIB)
    assert default_max_servers() == MIN_DERIVED_MAX_SERVERS
    monkeypatch.setattr("agent.lsp.manager._host_memory_bytes", lambda: 4096 * GIB)
    assert default_max_servers() == MAX_DERIVED_MAX_SERVERS


# ----------------------------------------------------------------------
# headline behaviour: the cap actually evicts a real process
# ----------------------------------------------------------------------


def test_population_cap_evicts_lru_and_kills_the_process(mock_roots):
    """THE test.  Three active roots, cap of two: the fleet must never
    hold three, and the evicted server's OS process must actually be
    gone — not merely dropped from a dict."""
    a, b, c = mock_roots("a"), mock_roots("b"), mock_roots("c")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,  # isolate the cap: no idle reaper involved
        max_servers=2,
    )
    try:
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(b))
        assert len(svc._clients) == 2

        a_pid = next(pid for key, pid in _pids(svc).items() if key[1] == str(a.parent))
        assert _alive(a_pid)

        # `a` is now the least-recently-used.  Spawning `c` must evict it.
        svc.get_diagnostics_sync(str(c))

        assert len(svc._clients) <= 2, "cap breached — fleet grew past max_servers"
        roots = {key[1] for key in svc._clients}
        assert str(a.parent) not in roots, "LRU root was not the one evicted"
        assert _wait_gone(a_pid), "evicted client was dropped from the dict but its process lives"
    finally:
        svc.shutdown()


def test_positive_control_uncapped_fleet_grows_to_three(mock_roots):
    """Positive control for the test above: identical scenario with the
    cap disabled must reach three live servers.  If this fails, the
    headline test proves nothing about the cap."""
    a, b, c = mock_roots("a"), mock_roots("b"), mock_roots("c")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,
        max_servers=0,  # disabled
    )
    try:
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(b))
        svc.get_diagnostics_sync(str(c))
        assert len(svc._clients) == 3, "control did not reproduce the unbounded fleet"
    finally:
        svc.shutdown()


def test_reuse_refreshes_lru_so_the_hot_root_survives(mock_roots):
    """A root served entirely from cache must not look progressively
    more idle — otherwise the cap evicts the busiest server."""
    a, b, c = mock_roots("a"), mock_roots("b"), mock_roots("c")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,
        max_servers=2,
    )
    try:
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(b))
        # Touch `a` again: `b` becomes the LRU.
        svc.get_diagnostics_sync(str(a))
        svc.get_diagnostics_sync(str(c))

        roots = {key[1] for key in svc._clients}
        assert str(a.parent) in roots, "the hot root was evicted instead of the cold one"
        assert str(b.parent) not in roots
    finally:
        svc.shutdown()


def test_cap_evicts_before_spawning_so_the_fleet_never_overshoots(mock_roots):
    """Eviction runs *before* the new spawn.  If it ran after, the fleet
    would transiently hold cap+1 servers — on a host already at its
    memory ceiling, the transient is the outage."""
    roots = [mock_roots(n) for n in ("a", "b", "c", "d", "e")]
    seen_peak = 0

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,
        max_servers=2,
    )
    try:
        for r in roots:
            svc.get_diagnostics_sync(str(r))
            seen_peak = max(seen_peak, len(svc._clients))
        assert seen_peak <= 2, f"fleet overshot the cap (peak {seen_peak})"
    finally:
        svc.shutdown()


def test_inflight_client_is_never_evicted(mock_roots):
    """A server draining a request must not be torn down under it.  The
    idle reaper is protected by MIN_IDLE_TIMEOUT exceeding the per-op
    wait budget; the cap has no such time guarantee, so it needs an
    explicit in-flight refcount."""
    a, b = mock_roots("a"), mock_roots("b")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0,
        max_servers=1,
    )
    try:
        svc.get_diagnostics_sync(str(a))
        key = next(iter(svc._clients))
        # Pin `a` as in-flight, then force a spawn that would evict it.
        svc._acquire(key)
        try:
            svc.get_diagnostics_sync(str(b))
            assert key in svc._clients, "cap evicted a client with an in-flight request"
        finally:
            svc._release(key)
    finally:
        svc.shutdown()


# ----------------------------------------------------------------------
# config wiring
# ----------------------------------------------------------------------


def test_max_servers_is_operator_configurable(monkeypatch):
    """An operator who cannot override the derived cap will disable LSP
    instead, which is a worse outcome than a tuned bound."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": False, "max_servers": 7}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._max_servers == 7
    finally:
        svc.shutdown()


def test_malformed_max_servers_falls_back_to_the_derived_default(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"lsp": {"enabled": False, "max_servers": "banana"}},
    )
    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._max_servers == default_max_servers()
    finally:
        svc.shutdown()
