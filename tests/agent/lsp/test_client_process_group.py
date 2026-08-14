"""Shutting a language server down must reap its whole process group.

The fleet cap counts *clients* -- one per (server_id, workspace root).
The memory does not live there.  ``typescript-language-server`` is a thin
supervisor that runs ``tsserver`` as a child, and ``tsserver`` is what
holds the project graph; ``pyright-langserver`` forks node workers the
same way.  If eviction signals only the supervisor's PID, the cap keeps
its promise on paper while the RSS it was introduced to bound walks away
as an orphan.

``LSPClient.start`` spawns with ``start_new_session=True``, so each
server is the leader of its *own* process group.  That is precisely what
makes group-directed signalling safe here: the group contains the server
and its workers and nothing else -- notably not the gateway or the TUI
parent, which is the accident the ``start_new_session`` comment in
``client.py`` exists to prevent.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX process groups; Windows kills via its own job-object path",
)


def _alive(pid: int) -> bool:
    """True while ``pid`` exists and has not been reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _await_death(pid: int, timeout: float = 5.0) -> bool:
    """Wait up to ``timeout`` for ``pid`` to disappear.

    Signal delivery and reaping are not instantaneous, so a bare
    post-shutdown ``_alive`` check would be racy in the *passing*
    direction.  Poll instead: a fixed sleep would either be flaky or slow.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _alive(pid)


@pytest.mark.live_system_guard_bypass
@pytest.mark.asyncio
async def test_shutdown_reaps_the_server_process_group(tmp_path: Path):
    """An evicted server's worker must not outlive it.

    Regression probe for the P1 raised on PR #52 ("Reap the entire
    language-server process group").  The finding was never ported to the
    live LSP stack, so closing #52 as superseded would have dropped it.
    """
    pidfile = tmp_path / "worker.pid"
    env = {
        "MOCK_LSP_SCRIPT": "spawns_child",
        "MOCK_LSP_CHILD_PIDFILE": str(pidfile),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    client = LSPClient(
        server_id="mock-spawns_child",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(tmp_path),
    )

    await client.start()
    server_pid = client._proc.pid  # noqa: SLF001 -- probing real process state

    # The worker is forked at startup, before the first message is served.
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        await asyncio.sleep(0.05)
    worker_pid = int(pidfile.read_text().strip())

    assert _alive(worker_pid), "mock worker never started; test proves nothing"
    assert os.getpgid(worker_pid) == os.getpgid(server_pid), (
        "worker escaped the server's process group -- this test would pass "
        "vacuously"
    )
    assert os.getpgid(server_pid) != os.getpgid(os.getpid()), (
        "server shares our process group; group-directed signals would hit "
        "the test runner"
    )

    try:
        await client.shutdown()

        assert await _await_death(server_pid), "supervisor survived shutdown"
        assert await _await_death(worker_pid), (
            f"worker {worker_pid} outlived its language server -- eviction "
            "signalled the supervisor PID only, so the memory the fleet cap "
            "exists to bound is still resident and now unowned"
        )
    finally:
        for pid in (worker_pid, server_pid):
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
