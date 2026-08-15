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
import signal
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from agent.lsp.client import _SIGKILL, LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX process groups; Windows kills via its own job-object path",
)


def _is_zombie(pid: int) -> bool:
    """True when ``pid`` has exited but nothing has reaped it yet."""
    try:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    state = out.stdout.strip()
    return bool(state) and state[0] == "Z"


def _alive(pid: int) -> bool:
    """True while ``pid`` is resident and can still run.

    A zombie answers ``os.kill(pid, 0)`` successfully, so signal-0 alone
    would call a successfully-killed worker a survivor.  That matters
    off this laptop: when the orphan is reparented to a PID 1 that does
    not promptly reap adopted children -- the default in a container --
    the corpse can linger past the poll deadline.  It holds no memory
    and cannot run, which is exactly what the assertion is asking about.
    """
    try:
        os.kill(pid, 0)  # windows-footgun: ok — module is skipped on win32
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid)


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


def test_signal_process_uses_the_pgid_captured_at_spawn():
    """The group must stay addressable after the supervisor is reaped.

    A cooperative server exits on our ``exit`` notification, so by the
    time cleanup runs its PID can already be gone.  Resolving the group
    then -- ``os.getpgid(proc.pid)`` -- raises, and the workers that are
    still holding the memory would never be signalled.  Capturing the
    pgid at spawn is what keeps them reachable.
    """
    signalled: list[tuple[int, int]] = []

    class _DeadSupervisor:
        """Its PID is unresolvable; only the saved pgid can save us."""

        pid = -1
        returncode = 0

        def terminate(self):
            signalled.append(("pid-only", signal.SIGTERM))

        def kill(self):
            signalled.append(("pid-only", _SIGKILL))

    client = LSPClient(
        server_id="mock",
        workspace_root="/tmp",
        command=["true"],
    )
    proc = _DeadSupervisor()

    with mock.patch.object(os, "killpg", lambda pgid, sig: signalled.append((pgid, sig))):
        client._signal_process(proc, signal.SIGTERM, 4242)  # noqa: SLF001

    assert signalled == [(4242, signal.SIGTERM)], (
        "cleanup fell back to the dead supervisor's PID instead of "
        "signalling the process group captured at spawn"
    )


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

    # Arm the cleanup guard the moment anything is running.  A missing or
    # unparseable pidfile, or a failed precondition, must not leak the
    # server and its worker onto the CI host for later tests to trip over.
    worker_pid = None
    try:
        # The worker is forked at startup, before the first message is
        # served, and publishes its pid only once SIGTERM is ignored.
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text(encoding="utf-8").strip():
                break
            await asyncio.sleep(0.05)
        assert pidfile.exists() and pidfile.read_text(encoding="utf-8").strip(), (
            "mock worker never published a pid; test proves nothing"
        )
        worker_pid = int(pidfile.read_text(encoding="utf-8").strip())

        assert _alive(worker_pid), "mock worker never started; test proves nothing"
        assert os.getpgid(worker_pid) == os.getpgid(server_pid), (
            "worker escaped the server's process group -- this test would pass "
            "vacuously"
        )
        assert os.getpgid(server_pid) != os.getpgid(os.getpid()), (
            "server shares our process group; group-directed signals would hit "
            "the test runner"
        )

        await client.shutdown()

        assert await _await_death(server_pid), "supervisor survived shutdown"
        assert await _await_death(worker_pid), (
            f"worker {worker_pid} outlived its language server -- eviction "
            "signalled the supervisor PID only, so the memory the fleet cap "
            "exists to bound is still resident and now unowned"
        )
    finally:
        for pid in (worker_pid, server_pid):
            if pid is None:
                continue
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
