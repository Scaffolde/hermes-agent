"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
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


def _install_mock_server(
    monkeypatch, script: str | list[str] = "errors", server_id: str = "pyright"
):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]
    scripts = [script] if isinstance(script, str) else script
    spawn_count = {"value": 0}

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        index = min(spawn_count["value"], len(scripts) - 1)
        spawn_count["value"] += 1
        env = {"MOCK_LSP_SCRIPT": scripts[index]}
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

    yield spawn_count

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


@pytest.mark.parametrize("failed_script", ["clean_eof", "malformed_frame"])
def test_service_replaces_client_after_reader_failure(
    tmp_path, monkeypatch, failed_script
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    source = repo / "x.py"
    source.write_text("print('hi')\n")
    monkeypatch.chdir(str(repo))
    server = _install_mock_server(
        monkeypatch, [failed_script, "clean"], "pyright"
    )
    spawn_count = next(server)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.5,
        install_strategy="manual",
    )
    try:
        async def _break_first_client():
            client = await svc._get_or_spawn(str(source))
            assert client is not None
            reader_task = client._reader_task
            assert reader_task is not None
            await client.open_file(str(source), language_id="python")
            await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)
            return client

        first = svc._loop.run(_break_first_client(), timeout=5.0)
        replacement = svc._loop.run(svc._get_or_spawn(str(source)), timeout=5.0)

        assert not first.is_running
        assert replacement is not None
        assert replacement is not first
        assert replacement.is_running
        assert spawn_count["value"] == 2
    finally:
        svc.shutdown()
        try:
            next(server)
        except StopIteration:
            pass


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


class _SteppedWallClock:
    """Stand-in for the ``time`` module with a stepped wall clock.

    ``time()`` is offset (an NTP correction); ``monotonic()`` is
    untouched, which is precisely the guarantee the idle bookkeeping is
    supposed to rely on.

    This models an NTP correction *only*.  Suspend/resume is a different
    failure mode with the opposite shape — see :class:`_SuspendedClock`.
    """

    def __init__(self, offset: float) -> None:
        self._offset = offset

    def time(self) -> float:
        return time.time() + self._offset

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class _SuspendedClock:
    """Stand-in for the ``time`` module across a Linux suspend/resume.

    Models what the kernel actually does, which is the inverse of an NTP
    step: ``CLOCK_MONOTONIC`` **stops** while the machine is suspended,
    while ``CLOCK_BOOTTIME`` keeps counting.  So after a resume the wall
    clock and BOOTTIME have both advanced by the suspend duration and
    ``monotonic()`` has not moved at all.

    ``has_boottime=False`` models a platform without ``CLOCK_BOOTTIME``
    (macOS), where the resolver must fall back to ``monotonic()``.
    """

    # Linux's real value; only identity across the two calls matters.
    _BOOTTIME_ID = 7

    def __init__(self, suspended_for: float, *, has_boottime: bool = True) -> None:
        self._suspended_for = suspended_for
        self._base_monotonic = time.monotonic()
        # Set per-instance so ``getattr(time, "CLOCK_BOOTTIME", None)``
        # genuinely misses on the no-BOOTTIME platform.
        if has_boottime:
            self.CLOCK_BOOTTIME = self._BOOTTIME_ID

    def time(self) -> float:
        return time.time() + self._suspended_for

    def monotonic(self) -> float:
        # Frozen: no awake time has elapsed since the fixture was built.
        return self._base_monotonic

    def clock_gettime(self, clk_id: int) -> float:
        if clk_id != self._BOOTTIME_ID:
            raise ValueError(f"unexpected clock id {clk_id}")
        return self._base_monotonic + self._suspended_for

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def test_idle_clock_counts_suspended_time(monkeypatch):
    """``_idle_clock`` must read a suspend-inclusive clock where one exists.

    ``CLOCK_MONOTONIC`` stops while the machine is suspended, so a laptop
    that sleeps longer than ``idle_timeout`` wakes with every ``_last_used``
    stamp still inside the window.  Reading ``CLOCK_BOOTTIME`` instead
    counts the suspended time and keeps the cutoff honest.

    Positive control: this fails against a ``time.monotonic()`` resolver,
    which returns the frozen base instead of the advanced value.
    """
    from agent.lsp import manager as manager_mod

    clock = _SuspendedClock(3600.0)
    monkeypatch.setattr(manager_mod, "time", clock)

    assert manager_mod._idle_clock() == pytest.approx(
        clock._base_monotonic + 3600.0
    ), "idle bookkeeping ignored suspended time — it must read CLOCK_BOOTTIME"


def test_idle_clock_falls_back_to_monotonic_without_boottime(monkeypatch):
    """No ``CLOCK_BOOTTIME`` (macOS) must degrade to ``monotonic``, not crash."""
    from agent.lsp import manager as manager_mod

    clock = _SuspendedClock(3600.0, has_boottime=False)
    monkeypatch.setattr(manager_mod, "time", clock)

    assert not hasattr(clock, "CLOCK_BOOTTIME")
    assert manager_mod._idle_clock() == pytest.approx(clock._base_monotonic)


def test_reaper_is_immune_to_suspend(mock_pyright, monkeypatch):
    """A suspend longer than ``idle_timeout`` must not stall the reaper.

    The inverse of the NTP case below: here the wall clock jumps *forward*
    and ``monotonic()`` freezes.  Against a ``monotonic()``-based cutoff the
    client looks freshly used no matter how long the machine slept, so the
    fleet accumulates across every sleep/wake cycle — exactly the leak the
    reaper exists to close.

    Positive control: this test fails against a ``monotonic()``-based
    reaper (the client survives the sweep) and passes against a
    BOOTTIME-based one.
    """
    from agent.lsp import manager as manager_mod

    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; the loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))

        # The client was last used "just now" — well inside the timeout.
        # Then the machine suspends for an hour, which advances BOOTTIME
        # but leaves monotonic exactly where it was.
        svc._last_used[key] = manager_mod._idle_clock()
        monkeypatch.setattr(manager_mod, "time", _SuspendedClock(3600.0))

        svc._loop.run(svc._reap_idle_once(), timeout=5.0)

        assert key not in svc._clients, (
            "an hour of suspend did not age the client — idle bookkeeping "
            "must count suspended time (CLOCK_BOOTTIME), not awake time only"
        )
        assert svc.get_status()["clients"] == []
    finally:
        svc.shutdown()


def test_reaper_is_immune_to_wall_clock_steps(mock_pyright, monkeypatch):
    """A backwards wall-clock step must not stall the idle reaper.

    Idle bookkeeping compares timestamps only to each other, so it must
    read a monotonic clock.  Against ``time.time()`` an NTP correction or
    a sleep/wake resume that steps the clock back further than
    ``idle_timeout`` drags the cutoff into the past, no client ever looks
    idle, and unbounded accumulation — the leak the reaper exists to
    close — silently comes back.

    Positive control: this test fails against a ``time.time()``-based
    reaper (the client survives the sweep) and passes against a
    monotonic one.
    """
    from agent.lsp import manager as manager_mod

    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; the loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))

        # Idle for longer than the timeout, in whichever clock is in use.
        svc._last_used[key] = svc._last_used[key] - 120.0

        # Now step the wall clock an hour into the past, mid-flight.
        monkeypatch.setattr(manager_mod, "time", _SteppedWallClock(-3600.0))

        svc._loop.run(svc._reap_idle_once(), timeout=5.0)

        assert key not in svc._clients, (
            "a backwards wall-clock step stalled the reaper — idle "
            "bookkeeping must read a monotonic clock"
        )
        assert svc.get_status()["clients"] == []
    finally:
        svc.shutdown()








