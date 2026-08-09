"""End-to-end fleet-cap coverage over the real spawn path.

:file:`test_client_cap.py` drives ``_enforce_cap_async`` directly against
hand-seeded state.  That proves the eviction *rule* — LRU victim, drain
until satisfied, skip in-flight — but it never exercises a spawn, so it
cannot see the thing the cap exists to bound: how many language servers
are alive on the host at once.

These tests go through ``get_diagnostics_sync`` and spawn real
subprocesses (the shared :file:`_mock_lsp_server.py`).  Every client is
freshly used throughout, so the idle reaper structurally cannot be what
holds the line — only the cap can.

That is the SCA-4389 incident shape: ~1.3 GiB per typescript-language-server
across many simultaneously-active worktrees on a 16 GiB host -> swap ->
free disk under the self-hosted runner's admission floor -> CI offline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


@pytest.fixture
def workspaces(tmp_path, monkeypatch):
    """Install the mock server under ``pyright`` and hand back a repo factory.

    Each call to the factory builds a *distinct* git workspace, so every
    file routes to its own ``(server_id, workspace_root)`` key and
    therefore its own client.  That is what lets these tests build a
    fleet at all.
    """
    index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": "errors"},
            initialization_options={},
        )

    SERVERS[index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock pyright (fleet-cap e2e)",
    )

    def make(name: str) -> Path:
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("")
        f = repo / "x.py"
        f.write_text("")
        return f

    # The workspace gate requires cwd to sit inside a git worktree.
    cwd = tmp_path / "cwd_repo"
    cwd.mkdir()
    (cwd / ".git").mkdir()
    monkeypatch.chdir(str(cwd))

    yield make

    SERVERS[index] = original


def _service(max_clients: int) -> LSPService:
    return LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=600.0,  # long: nothing is idle-reapable during these tests
        max_clients=max_clients,
    )


def _running(svc: LSPService) -> int:
    return sum(1 for c in svc._clients.values() if c.is_running)


def test_population_never_exceeds_the_cap_during_a_spawn(workspaces, monkeypatch):
    """Room must be made *before* the spawn, not after it.

    Evicting after ``client.start()`` means the fleet momentarily holds
    ``cap + 1`` live servers.  On a host already at its memory ceiling
    that transient is not a detail — it *is* the outage, because the
    extra ~1.3 GiB is resident before anything is reclaimed.

    Probe: record how many clients are already running each time a new
    server is about to start.  Making room first bounds that at
    ``cap - 1``; making room afterwards lets it reach ``cap``.
    """
    cap = 2
    svc = _service(max_clients=cap)
    observed: list[int] = []

    real_start = LSPClient.start

    async def counting_start(self):
        observed.append(_running(svc))
        return await real_start(self)

    monkeypatch.setattr(LSPClient, "start", counting_start)

    try:
        for name in ("a", "b", "c", "d"):
            svc.get_diagnostics_sync(str(workspaces(name)))

        assert observed, "no server was ever spawned — the fixture is not wired"
        assert max(observed) <= cap - 1, (
            f"a spawn began while {max(observed)} servers were already live "
            f"under cap={cap}; the fleet peaked at {max(observed) + 1}. "
            "Evict before spawning, not after."
        )
    finally:
        svc.shutdown()


def test_cap_holds_the_fleet_with_every_client_active(workspaces):
    """N+1 *active* clients must settle at N.

    Nothing here is idle — every client was used seconds ago, so
    ``_reap_idle_once`` would evict none of them.  This is the peak case
    the reaper structurally cannot cover.
    """
    cap = 2
    svc = _service(max_clients=cap)
    try:
        for name in ("a", "b", "c"):
            svc.get_diagnostics_sync(str(workspaces(name)))

        assert len(svc._clients) == cap, (
            f"cap={cap} must hold the fleet at {cap}, got {len(svc._clients)}"
        )

        # Prove the reaper could not have produced this result.
        cutoff = time.time() - svc._idle_timeout
        assert all(ts > cutoff for ts in svc._last_used.values()), (
            "every surviving client must be non-idle, otherwise this test "
            "proves nothing the idle reaper does not already cover"
        )
    finally:
        svc.shutdown()


def test_the_evicted_server_process_actually_dies(workspaces):
    """Eviction must reclaim the process.

    Dropping the client from the map while its server keeps running
    would make the cap a bookkeeping lie and leave in place exactly the
    memory it exists to bound.
    """
    svc = _service(max_clients=1)
    try:
        svc.get_diagnostics_sync(str(workspaces("a")))
        victim = next(iter(svc._clients.values()))
        assert victim.is_running

        svc.get_diagnostics_sync(str(workspaces("b")))

        assert victim not in svc._clients.values(), "victim must leave the map"
        deadline = time.time() + 5.0
        while victim.is_running and time.time() < deadline:
            time.sleep(0.05)
        assert not victim.is_running, "the evicted server process must be stopped"
    finally:
        svc.shutdown()


def test_the_least_recently_used_workspace_is_the_victim(workspaces):
    """The victim must be the LRU client, not an arbitrary one."""
    svc = _service(max_clients=2)
    try:
        fa, fb = workspaces("a"), workspaces("b")
        svc.get_diagnostics_sync(str(fa))
        svc.get_diagnostics_sync(str(fb))

        key_a = next(k for k in svc._clients if str(fa.parent) in k[1])
        key_b = next(k for k in svc._clients if str(fb.parent) in k[1])

        # Make A unambiguously the least-recently-used.
        svc._last_used[key_a] = time.time() - 120.0
        svc._last_used[key_b] = time.time()

        svc.get_diagnostics_sync(str(workspaces("c")))

        assert key_a not in svc._clients, "LRU client A must be the victim"
        assert key_b in svc._clients, "more-recently-used client B must survive"
        assert key_a not in svc._last_used, (
            "an evicted key must not orphan its _last_used entry"
        )
    finally:
        svc.shutdown()


def test_the_idle_reaper_still_works_under_the_cap(workspaces):
    """SCA-4620 anti-criterion.

    Upstream owns idle reaping (NousResearch ``d7578018c`` / ``24a56f027``);
    re-implementing it is what made fork PRs #50/#51/#52 unmergeable.  The
    cap sits *on top of* the reaper, so the reaper must still be running
    and still do its own job.
    """
    svc = _service(max_clients=8)
    try:
        svc.get_diagnostics_sync(str(workspaces("a")))
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()

        key = next(iter(svc._clients))
        svc._last_used[key] = time.time() - (svc._idle_timeout + 60.0)
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)

        assert key not in svc._clients, "idle reaping must still work under the cap"
    finally:
        svc.shutdown()


def test_a_concurrent_spawn_does_not_evict_a_client_mid_handover(workspaces, monkeypatch):
    """A live server must survive another root's sweep during handover.

    ``_get_or_spawn`` publishes into ``_clients`` and only then returns;
    the caller runs ``_acquire`` after that return.  Across that window
    the client is a running subprocess whose in-flight count is still
    zero.  A concurrent spawn for a *different* root sweeps with no
    ``protect`` — ``protect`` is per-call and names only the sweeping
    caller's own key — so before the fix it selected this client as its
    LRU victim and shut the process down.  The original caller then got
    back a dead client and silently lost every diagnostic.

    The window is reproduced exactly where it occurs: inside the
    protect-sweep, which is the one call that provably runs after the
    insert and before ``_spawning`` clears.
    """
    cap = 1
    svc = _service(max_clients=cap)
    real_enforce = svc._enforce_cap_async
    other = ("pyright", "/a-concurrently-spawning-root")
    ran: list[bool] = []

    async def enforcing(protect=None):
        if protect is not None and not ran:
            ran.append(True)
            # Stand in for a second root that has reserved its slot and
            # is sweeping to make room for itself.
            svc._spawning[other] = None
            try:
                await real_enforce()
            finally:
                svc._spawning.pop(other, None)
        return await real_enforce(protect=protect)

    monkeypatch.setattr(svc, "_enforce_cap_async", enforcing)
    try:
        svc.get_diagnostics_sync(str(workspaces("a")))

        assert ran, "the handover window never ran — the test is not wired"
        assert len(svc._clients) == 1, (
            "the concurrent sweep evicted a client that had not reached "
            "its caller yet"
        )
        client = next(iter(svc._clients.values()))
        assert client.is_running, (
            "the client survived in the map but its process was shut "
            "down — the caller holds a dead server"
        )
    finally:
        svc.shutdown()
