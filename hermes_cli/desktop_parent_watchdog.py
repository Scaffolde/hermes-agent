"""Desktop-owned dashboard backend parent watchdog.

The Electron app spawns ``hermes dashboard --port 0`` as a local backend. If
Electron crashes or is force-quit, Python is re-parented to the OS service
manager and keeps serving a stale, token-protected dashboard port. That orphan can
burn CPU and leave the next desktop launch talking to a dead/stale backend.

This module is intentionally tiny so it can start early in the ``dashboard``
command without dragging in the web server. ``psutil`` is a core Hermes
dependency and is used for cross-platform PID liveness; do not use
``os.kill(pid, 0)`` here because that sends CTRL_C_EVENT on Windows.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Optional

_LOG = logging.getLogger(__name__)
_ORPHAN_PARENT_PIDS = {0, 1}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_WINDOWS_PARENT_START_EPOCH_TOLERANCE_S = 5.0


def _parse_parent_pid(raw: object) -> Optional[int]:
    try:
        pid = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _parse_parent_start_epoch(raw: object) -> Optional[float]:
    try:
        epoch = float(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    return epoch if epoch > 0 else None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except ImportError:
        # psutil is a core dependency. If a stripped install is missing it,
        # fail safe toward keeping the backend alive rather than using
        # platform-specific signal probes here. In particular, os.kill(pid, 0)
        # is not a harmless liveness check on Windows.
        return True


def _process_create_time(pid: int) -> Optional[float]:
    if pid <= 0:
        return None
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except ImportError:
        # psutil is a core dependency. On Windows, parent identity checking is
        # specifically what prevents PID reuse from keeping an orphan alive, so
        # callers treat an unknown create_time as a failed identity check.
        return None
    except Exception:
        return None


def _windows_parent_identity_matches(
    parent_pid: int,
    expected_parent_start_epoch: float,
    *,
    process_create_time: Callable[[int], Optional[float]] = _process_create_time,
    tolerance_s: float = _WINDOWS_PARENT_START_EPOCH_TOLERANCE_S,
) -> bool:
    actual = process_create_time(parent_pid)
    if actual is None:
        return False
    return abs(actual - expected_parent_start_epoch) <= tolerance_s


def _should_exit_for_parent(
    parent_pid: int,
    *,
    expected_parent_start_epoch: float | None = None,
    getppid: Callable[[], int] = os.getppid,
    pid_exists: Callable[[int], bool] = _pid_exists,
    process_create_time: Callable[[int], Optional[float]] = _process_create_time,
    platform: str = sys.platform,
) -> bool:
    """Return True when the desktop parent is gone.

    Prefer the direct parent relationship: while Electron owns this backend,
    ``os.getppid()`` equals the PID Electron passed in. If the process is
    re-parented to launchd/init (0/1), the desktop is gone even if the old PID
    has already been recycled. For unusual launch wrappers where the immediate
    parent differs but the Electron PID still exists, stay alive.
    """

    current_parent = getppid()
    if not pid_exists(parent_pid):
        return True
    if platform == "win32":
        if expected_parent_start_epoch is None:
            return True
        if not _windows_parent_identity_matches(
            parent_pid,
            expected_parent_start_epoch,
            process_create_time=process_create_time,
        ):
            return True
    if current_parent == parent_pid:
        return False
    if current_parent in _ORPHAN_PARENT_PIDS:
        return True
    return False


def start_desktop_parent_watchdog(
    env: Mapping[str, str] | None = None,
    *,
    interval_s: float = 5.0,
    exit_fn: Callable[[int], None] = os._exit,
) -> threading.Thread | None:
    """Start a daemon thread that exits when the Electron parent disappears.

    Only enabled for desktop-spawned dashboard backends that set both
    ``HERMES_DESKTOP=1`` and ``HERMES_DESKTOP_PARENT_PID``. Returns the thread in
    tests/diagnostics and ``None`` when disabled or misconfigured.
    """

    env = env or os.environ
    if str(env.get("HERMES_DESKTOP", "")).strip().lower() not in _TRUE_VALUES:
        return None

    parent_pid = _parse_parent_pid(env.get("HERMES_DESKTOP_PARENT_PID"))
    if parent_pid is None:
        return None
    expected_parent_start_epoch = _parse_parent_start_epoch(env.get("HERMES_DESKTOP_PARENT_START_EPOCH"))

    def _watch() -> None:
        while True:
            time.sleep(max(0.25, interval_s))
            if _should_exit_for_parent(parent_pid, expected_parent_start_epoch=expected_parent_start_epoch):
                _LOG.warning(
                    "Desktop parent PID %s disappeared or no longer matches; exiting orphaned dashboard backend",
                    parent_pid,
                )
                exit_fn(0)
                return

    thread = threading.Thread(
        target=_watch,
        name="desktop-parent-watchdog",
        daemon=True,
    )
    thread.start()
    return thread
