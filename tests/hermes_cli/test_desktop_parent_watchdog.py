from __future__ import annotations

import threading

import hermes_cli.desktop_parent_watchdog as watchdog
from hermes_cli.desktop_parent_watchdog import (
    _parse_parent_pid,
    _parse_parent_start_epoch,
    _should_exit_for_parent,
    start_desktop_parent_watchdog,
)


def test_parse_parent_pid_accepts_positive_ints_only():
    assert _parse_parent_pid("123") == 123
    assert _parse_parent_pid(456) == 456
    assert _parse_parent_pid("0") is None
    assert _parse_parent_pid("-1") is None
    assert _parse_parent_pid("not-a-pid") is None
    assert _parse_parent_pid(None) is None


def test_parse_parent_start_epoch_accepts_positive_epoch_seconds_only():
    assert _parse_parent_start_epoch("1780000000.25") == 1780000000.25
    assert _parse_parent_start_epoch(1780000000) == 1780000000.0
    assert _parse_parent_start_epoch("0") is None
    assert _parse_parent_start_epoch("-1") is None
    assert _parse_parent_start_epoch("not-a-time") is None
    assert _parse_parent_start_epoch(None) is None


def test_should_exit_only_after_desktop_parent_is_gone():
    assert _should_exit_for_parent(4242, getppid=lambda: 4242, pid_exists=lambda _pid: True) is False
    # Windows can retain the creator PID in getppid() after that PID exits.
    assert _should_exit_for_parent(4242, getppid=lambda: 4242, pid_exists=lambda _pid: False) is True
    assert _should_exit_for_parent(4242, getppid=lambda: 1, pid_exists=lambda _pid: True) is True
    assert _should_exit_for_parent(4242, getppid=lambda: 99, pid_exists=lambda _pid: False) is True
    # Wrapper/shell launch case: immediate parent differs, but the desktop PID is
    # still alive, so do not self-kill immediately.
    assert _should_exit_for_parent(4242, getppid=lambda: 99, pid_exists=lambda _pid: True) is False


def test_windows_parent_identity_uses_create_time_to_reject_reused_pid():
    assert (
        _should_exit_for_parent(
            4242,
            expected_parent_start_epoch=1780000000.0,
            platform="win32",
            getppid=lambda: 4242,
            pid_exists=lambda _pid: True,
            process_create_time=lambda _pid: 1780000001.5,
        )
        is False
    )

    assert (
        _should_exit_for_parent(
            4242,
            expected_parent_start_epoch=1780000000.0,
            platform="win32",
            getppid=lambda: 4242,
            pid_exists=lambda _pid: True,
            process_create_time=lambda _pid: 1780000015.0,
        )
        is True
    )

    assert (
        _should_exit_for_parent(
            4242,
            expected_parent_start_epoch=None,
            platform="win32",
            getppid=lambda: 4242,
            pid_exists=lambda _pid: True,
            process_create_time=lambda _pid: 1780000000.0,
        )
        is True
    )


def test_non_windows_parent_watchdog_ignores_create_time_identity_check():
    assert (
        _should_exit_for_parent(
            4242,
            expected_parent_start_epoch=1780000000.0,
            platform="darwin",
            getppid=lambda: 4242,
            pid_exists=lambda _pid: True,
            process_create_time=lambda _pid: 1780000015.0,
        )
        is False
    )


def test_watchdog_disabled_without_desktop_parent_env():
    assert start_desktop_parent_watchdog({"HERMES_DESKTOP": "1"}) is None
    assert start_desktop_parent_watchdog({"HERMES_DESKTOP_PARENT_PID": "123"}) is None
    assert start_desktop_parent_watchdog({"HERMES_DESKTOP": "1", "HERMES_DESKTOP_PARENT_PID": "bad"}) is None


def test_watchdog_thread_invokes_exit_after_parent_disappears(monkeypatch):
    exited = threading.Event()
    monkeypatch.setattr(watchdog, "_should_exit_for_parent", lambda _pid, **_kwargs: True)

    thread = start_desktop_parent_watchdog(
        {"HERMES_DESKTOP": "1", "HERMES_DESKTOP_PARENT_PID": "123"},
        interval_s=0,
        exit_fn=lambda _code: exited.set(),
    )

    assert thread is not None
    assert exited.wait(timeout=1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_watchdog_passes_parent_start_epoch_from_env(monkeypatch):
    seen = {}
    exited = threading.Event()

    def fake_should_exit(pid, **kwargs):
        seen["pid"] = pid
        seen.update(kwargs)
        return True

    monkeypatch.setattr(watchdog, "_should_exit_for_parent", fake_should_exit)

    thread = start_desktop_parent_watchdog(
        {
            "HERMES_DESKTOP": "1",
            "HERMES_DESKTOP_PARENT_PID": "123",
            "HERMES_DESKTOP_PARENT_START_EPOCH": "1780000000.25",
        },
        interval_s=0,
        exit_fn=lambda _code: exited.set(),
    )

    assert thread is not None
    assert exited.wait(timeout=1)
    thread.join(timeout=1)
    assert seen["pid"] == 123
    assert seen["expected_parent_start_epoch"] == 1780000000.25
