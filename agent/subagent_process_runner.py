"""Host-owned subprocess runner for isolated Hermes subagent execution.

Portable mode owns a POSIX process group, scrubs the environment, bounds I/O,
and reaps the group, but deliberately claims no OS confinement.  Linux strict
mode adds a fail-closed bubblewrap + delegated cgroup-v2 backend.  This module
does not import lifecycle, profile, broker, or receipt implementations; the
small callback protocols below are the serial integration seam.
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import secrets
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from agent.subagent_worker_main import send_capability_secret, send_authenticated_frame

PORTABLE_CONFINEMENT = "portable-process-unconfined"
LINUX_STRICT_CONFINEMENT = "linux-strict-bwrap-cgroup-v2"
_VALID_BACKENDS = frozenset({"portable", "linux-strict"})
_MAX_ARG_BYTES = 128_000
_DEFAULT_OUTPUT_BYTES = 1_048_576
_STRICT_WORKER_HOME = Path("/tmp/hermes-worker-home")
_BROKER_CONTAINMENT_FAIL_STOP_SECONDS = 2.0
_BROKER_CONTAINMENT_EXIT_CODE = 70
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


class _BrokerContainmentFailStop(BaseException):
    """Non-recoverable fallback when the process-level fail-stop unexpectedly returns."""


def _bounded_diagnostic_text(value: object, *, limit: int) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(value)
    )
    return " ".join(printable.split())[:limit]


def _fail_stop_broker_containment(reason: str) -> None:
    """Terminate Hermes before any receipt can follow an uncontained host effect."""

    del reason
    os._exit(_BROKER_CONTAINMENT_EXIT_CODE)


class BrokerCallbacks(Protocol):
    """Host broker adapter invoked on a bounded inherited socket."""

    def serve(
        self,
        channel: socket.socket,
        *,
        root_pid: int,
        stop_requested: threading.Event,
    ) -> None: ...

    def cancel(self) -> bool | None:
        """Revoke accepted host operations, reporting whether they quiesced."""
        ...


class ReceiptCallbacks(Protocol):
    """Receipt adapter kept independent from Lane A's concrete state machine."""

    def on_created(self, *, backend: str, confinement: str) -> None: ...

    def on_started(self, *, root_pid: int) -> None: ...

    def on_terminal(self, *, state: str, result: "ProcessRunResult") -> None: ...


@dataclasses.dataclass(frozen=True)
class RuntimeMount:
    source: Path
    target: Path


@dataclasses.dataclass(frozen=True)
class LinuxStrictConfig:
    cgroup_parent: Path
    bwrap_path: str | None = None
    sandbox_workspace: Path = Path("/workspace")
    allow_network: bool = False
    namespace_probe_timeout_seconds: float = 3.0


@dataclasses.dataclass(frozen=True)
class StrictPrerequisiteResult:
    available: bool
    diagnostics: tuple[str, ...] = ()
    bwrap_path: str | None = None


@dataclasses.dataclass(frozen=True)
class CleanupEvidence:
    requested: bool = True
    descendant_scope: str = "process-group"
    term_sent: bool = False
    kill_sent: bool = False
    root_reaped: bool = False
    process_group_empty: bool = False
    cgroup_kill_sent: bool = False
    cgroup_empty: bool | None = None
    cgroup_removed: bool | None = None
    broker_quiesced: bool | None = None
    broker_active_operation: str | None = None


@dataclasses.dataclass(frozen=True)
class ProcessRunResult:
    backend: str
    confinement: str
    state: str
    root_pid: int | None
    returncode: int | None
    exit_code: int | None
    signal: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    cleanup: CleanupEvidence
    diagnostic: str | None = None


@dataclasses.dataclass(frozen=True)
class ProcessRunSpec:
    executable: str
    argv: tuple[str, ...]
    cwd: Path
    workspace: Path
    capability_secret: bytes = dataclasses.field(repr=False)
    capability_id: str = dataclasses.field(default="runner-capability", repr=False)
    launch_receipt_digest: str = dataclasses.field(default="0" * 64, repr=False)
    backend: Literal["portable", "linux-strict"] = "portable"
    capability_fd_arg: str | None = "--capability-fd"
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    ambient_env_allowlist: tuple[str, ...] = ()
    runtime_mounts: tuple[RuntimeMount, ...] = ()
    strict: LinuxStrictConfig | None = None
    timeout_seconds: float = 300.0
    term_grace_seconds: float = 2.0
    kill_grace_seconds: float = 2.0
    max_output_bytes: int = _DEFAULT_OUTPUT_BYTES
    cancellation_event: threading.Event | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    broker: BrokerCallbacks | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    receipt: ReceiptCallbacks | None = dataclasses.field(
        default=None, repr=False, compare=False
    )


def _forbidden_environment_name(name: str) -> bool:
    upper = name.upper()
    if upper.startswith(("PYTHON", "LD_", "DYLD_")):
        return True
    if upper in {
        "ALL_PROXY",
        "AWS_CONFIG_FILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_PROFILE",
        "AZURE_CONFIG_DIR",
        "BASH_ENV",
        "CLOUDSDK_CONFIG",
        "DOCKER_CONFIG",
        "ENV",
        "GIT_ASKPASS",
        "KUBECONFIG",
        "NETRC",
        "NO_PROXY",
        "SSH_AUTH_SOCK",
    }:
        return True
    if upper.endswith("_PROXY"):
        return True
    if upper.startswith("HERMES_"):
        return True
    secret_markers = (
        "API_KEY",
        "ACCESS_KEY",
        "SECRET",
        "TOKEN",
        "PASSWORD",
        "CREDENTIAL",
        "PRIVATE_KEY",
    )
    return any(marker in upper for marker in secret_markers)


def _validate_environment_entry(name: Any, value: Any) -> tuple[str, str]:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "=" in name
        or not isinstance(value, str)
        or "\x00" in value
    ):
        raise ValueError("environment entries must be NUL-free strings")
    if _forbidden_environment_name(name):
        raise ValueError(f"forbidden environment variable: {name}")
    return name, value


def _minimal_environment(
    *,
    explicit: Mapping[str, str] | None = None,
    ambient_allowlist: Sequence[str] = (),
) -> dict[str, str]:
    """Build a fresh environment; ambient state is never copied wholesale."""

    result: dict[str, str] = {}
    for name in ambient_allowlist:
        if not isinstance(name, str) or _forbidden_environment_name(name):
            continue
        value = os.environ.get(name)
        if value is not None:
            key, clean_value = _validate_environment_entry(name, value)
            result[key] = clean_value
    for name, value in (explicit or {}).items():
        key, clean_value = _validate_environment_entry(name, value)
        result[key] = clean_value
    return result


def _path_is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _validate_argv(executable: str, argv: Sequence[str]) -> None:
    if not isinstance(executable, str) or not os.path.isabs(executable):
        raise ValueError("executable must be an absolute path")
    values = (executable, *argv)
    if any(not isinstance(value, str) or "\x00" in value for value in values):
        raise ValueError("argv must contain only NUL-free strings")
    if sum(len(value.encode("utf-8")) + 1 for value in values) > _MAX_ARG_BYTES:
        raise ValueError("argv exceeds the configured byte bound")


def _validate_spec(spec: ProcessRunSpec) -> None:
    if os.name != "posix":
        raise ValueError("owned process runner currently requires POSIX process groups")
    if spec.backend not in _VALID_BACKENDS:
        raise ValueError(f"unknown process backend: {spec.backend}")
    _validate_argv(spec.executable, spec.argv)
    if (
        not isinstance(spec.capability_secret, bytes)
        or not 32 <= len(spec.capability_secret) <= 128
    ):
        raise ValueError("capability_secret must contain 32-128 bytes")
    if not spec.capability_id or not spec.launch_receipt_digest:
        raise ValueError("capability bootstrap authority is required")
    if spec.broker is not None and not callable(getattr(spec.broker, "cancel", None)):
        raise ValueError("broker must provide synchronous cancellation")
    if spec.capability_fd_arg is not None and (
        not spec.capability_fd_arg or "\x00" in spec.capability_fd_arg
    ):
        raise ValueError("capability_fd_arg must be a safe non-empty string")
    for name, value in (
        ("timeout_seconds", spec.timeout_seconds),
        ("term_grace_seconds", spec.term_grace_seconds),
        ("kill_grace_seconds", spec.kill_grace_seconds),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(spec.max_output_bytes, int) or spec.max_output_bytes < 0:
        raise ValueError("max_output_bytes must be a non-negative integer")
    if not spec.workspace.is_absolute() or not spec.cwd.is_absolute():
        raise ValueError("workspace and cwd must be absolute")
    if not spec.workspace.is_dir() or not spec.cwd.is_dir():
        raise ValueError("workspace and cwd must exist as directories")
    if not _path_is_beneath(spec.cwd, spec.workspace):
        raise ValueError("cwd must resolve beneath workspace")
    _minimal_environment(
        explicit=spec.environment,
        ambient_allowlist=spec.ambient_env_allowlist,
    )
    if spec.backend == "portable":
        if spec.strict is not None or spec.runtime_mounts:
            raise ValueError("portable backend does not accept strict backend options")
        if not Path(spec.executable).is_file():
            raise ValueError("portable executable must be an existing file")
    elif spec.strict is None:
        raise ValueError("linux-strict backend requires strict configuration")


def _probe_cgroup_delegation(parent: Path) -> str | None:
    if not parent.is_absolute() or not parent.is_dir():
        return "delegated cgroup parent is not an existing absolute directory"
    if not (parent / "cgroup.controllers").is_file():
        return "delegated cgroup parent is not cgroup v2"
    probe = parent / f".hermes-probe-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        probe.mkdir(mode=0o700)
        procs = probe / "cgroup.procs"
        fd = os.open(procs, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
        os.close(fd)
    except OSError as exc:
        return f"delegated cgroup subtree is not writable: errno={exc.errno}"
    finally:
        try:
            probe.rmdir()
        except OSError:
            pass
    return None


def _probe_namespace_support(bwrap_path: str, timeout_seconds: float) -> str | None:
    argv = [
        bwrap_path,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--",
        "/bin/true",
    ]
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            env={},
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"required namespace probe failed: {type(exc).__name__}"
    if completed.returncode != 0:
        return f"required namespace probe exited {completed.returncode}"
    return None


def probe_linux_strict(config: LinuxStrictConfig) -> StrictPrerequisiteResult:
    """Prove every strict prerequisite before any worker process is spawned."""

    diagnostics: list[str] = []
    if platform.system() != "Linux":
        diagnostics.append("linux-strict backend requires Linux")
        return StrictPrerequisiteResult(False, tuple(diagnostics))
    bwrap_path = config.bwrap_path or shutil.which("bwrap")
    if (
        not bwrap_path
        or not os.path.isabs(bwrap_path)
        or not os.access(bwrap_path, os.X_OK)
    ):
        diagnostics.append("bwrap is unavailable or not executable")
    cgroup_diagnostic = _probe_cgroup_delegation(config.cgroup_parent)
    if cgroup_diagnostic:
        diagnostics.append(cgroup_diagnostic)
    if bwrap_path and not diagnostics:
        namespace_diagnostic = _probe_namespace_support(
            bwrap_path, config.namespace_probe_timeout_seconds
        )
        if namespace_diagnostic:
            diagnostics.append(namespace_diagnostic)
    return StrictPrerequisiteResult(
        available=not diagnostics,
        diagnostics=tuple(diagnostics),
        bwrap_path=bwrap_path if not diagnostics else None,
    )


def _sandbox_path_is_safe(path: Path) -> bool:
    raw = str(path)
    return (
        path.is_absolute()
        and raw != "/"
        and "\x00" not in raw
        and all(segment not in {"", ".", ".."} for segment in raw.split("/")[1:])
    )


def _mount_parent_directives(targets: Sequence[Path]) -> list[str]:
    parents: set[str] = set()
    reserved = {"/tmp", "/proc", "/dev", "/workspace"}
    for target in targets:
        parent = target.parent
        while str(parent) not in {"/", "."}:
            rendered = str(parent)
            if rendered not in reserved:
                parents.add(rendered)
            parent = parent.parent
    argv: list[str] = []
    for parent in sorted(parents, key=lambda value: (value.count("/"), value)):
        argv.extend(("--dir", parent))
    return argv


def _build_linux_strict_argv(
    spec: ProcessRunSpec,
    bwrap_path: str,
    *,
    capability_fd: int,
) -> list[str]:
    """Build bubblewrap argv without shell parsing or implicit host mounts."""

    strict = spec.strict
    if strict is None:
        raise ValueError("strict configuration is required")
    sandbox_workspace = strict.sandbox_workspace
    if not _sandbox_path_is_safe(sandbox_workspace):
        raise ValueError("sandbox workspace path is unsafe")
    seen_targets: set[str] = set()
    for mount in spec.runtime_mounts:
        if not mount.source.is_absolute() or not mount.source.exists():
            raise ValueError("runtime mount source must be an existing absolute path")
        if not _sandbox_path_is_safe(mount.target):
            raise ValueError("runtime mount target is unsafe")
        target = str(mount.target)
        if target in seen_targets or target in {
            "/tmp",
            "/proc",
            "/dev",
            str(sandbox_workspace),
        }:
            raise ValueError("runtime mount target is duplicate or reserved")
        seen_targets.add(target)
    environment = _minimal_environment(
        explicit=spec.environment,
        ambient_allowlist=spec.ambient_env_allowlist,
    )
    environment["HOME"] = str(_STRICT_WORKER_HOME)
    argv = [
        bwrap_path,
        # Bubblewrap passes otherwise-unreferenced inherited descriptors to the
        # sandbox command. Popen.pass_fds constrains that set to the capability
        # socket (the cgroup launcher fd is closed before bwrap is exec'd).
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--clearenv",
    ]
    if not strict.allow_network:
        argv.append("--unshare-net")
    for name, value in sorted(environment.items()):
        argv.extend(("--setenv", name, value))
    argv.extend(("--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev"))
    argv.extend(
        _mount_parent_directives(
            [mount.target for mount in spec.runtime_mounts] + [sandbox_workspace]
        )
    )
    argv.extend(("--dir", str(_STRICT_WORKER_HOME)))
    argv.extend(("--dir", str(sandbox_workspace)))
    argv.extend(("--bind", str(spec.workspace), str(sandbox_workspace)))
    # A runtime file may intentionally overlay a path inside the writable
    # workspace (for example a host-owned benchmark authority snapshot). Apply
    # runtime mounts after the workspace bind so that bind cannot shadow them.
    for mount in spec.runtime_mounts:
        argv.extend(("--ro-bind", str(mount.source), str(mount.target)))
    relative_cwd = spec.cwd.resolve(strict=True).relative_to(
        spec.workspace.resolve(strict=True)
    )
    sandbox_cwd = sandbox_workspace / relative_cwd
    argv.extend(("--chdir", str(sandbox_cwd), "--", spec.executable, *spec.argv))
    if spec.capability_fd_arg is not None:
        argv.extend((spec.capability_fd_arg, str(capability_fd)))
    _validate_argv(argv[0], argv[1:])
    return argv


def _create_owned_cgroup(parent: Path) -> tuple[Path, int]:
    cgroup = parent / f"hermes-evo-{os.getpid()}-{secrets.token_hex(8)}"
    cgroup.mkdir(mode=0o700)
    try:
        fd = os.open(
            cgroup / "cgroup.procs",
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        try:
            cgroup.rmdir()
        except OSError:
            pass
        raise
    return cgroup, fd


def _read_cgroup_pids(cgroup: Path) -> tuple[int, ...]:
    values = (cgroup / "cgroup.procs").read_text(encoding="ascii").splitlines()
    pids: list[int] = []
    for value in values:
        try:
            pid = int(value)
        except ValueError:
            continue
        if pid > 0:
            pids.append(pid)
    return tuple(pids)


def cleanup_cgroup(cgroup: Path, *, timeout_seconds: float) -> CleanupEvidence:
    """Kill only members of the dedicated cgroup and prove it became empty."""

    deadline = time.monotonic() + timeout_seconds
    kill_sent = False
    readable = True
    while True:
        try:
            pids = _read_cgroup_pids(cgroup)
        except OSError:
            readable = False
            break
        if not pids:
            break
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, _SIGKILL)
                kill_sent = True
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    try:
        empty = readable and not _read_cgroup_pids(cgroup)
    except OSError:
        empty = False
    removed = False
    if empty:
        try:
            cgroup.rmdir()
            removed = True
        except OSError:
            pass
    return CleanupEvidence(
        descendant_scope="cgroup-v2",
        cgroup_kill_sent=kill_sent,
        cgroup_empty=empty,
        cgroup_removed=removed,
    )


def _process_group_exists(pgid: int) -> bool:
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        return False
    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(pgid: int, sig: signal.Signals) -> bool:
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        return False
    try:
        killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


def _append_bounded(buffer: bytearray, chunk: bytes, cap: int) -> bool:
    remaining = max(0, cap - len(buffer))
    buffer.extend(chunk[:remaining])
    return len(chunk) > remaining


def _monitor_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    term_grace_seconds: float,
    kill_grace_seconds: float,
    max_output_bytes: int,
    cancellation_event: threading.Event | None = None,
) -> tuple[bytes, bytes, bool, bool, bool, bool]:
    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = False
    stderr_truncated = False
    selector = selectors.DefaultSelector()
    streams: dict[int, tuple[str, Any]] = {}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        fd = stream.fileno()
        os.set_blocking(fd, False)
        selector.register(fd, selectors.EVENT_READ)
        streams[fd] = (name, stream)

    started = time.monotonic()
    deadline = started + timeout_seconds
    term_deadline: float | None = None
    kill_deadline: float | None = None
    timed_out = False
    term_sent = False
    kill_sent = False
    term_attempted = False
    kill_attempted = False
    cancellation_observed = False
    terminal_trigger: Literal["timeout", "cancellation"] | None = None
    group_id = process.pid
    drain_deadline: float | None = None
    try:
        while True:
            now = time.monotonic()
            returncode = process.poll()
            group_exists = _process_group_exists(group_id)
            cancellation_requested = bool(
                cancellation_event is not None and cancellation_event.is_set()
            )
            if (
                terminal_trigger is None
                and cancellation_requested
                and now < deadline
                and (returncode is None or group_exists)
            ):
                terminal_trigger = "cancellation"
                cancellation_observed = True
            if returncode is None and now >= deadline and terminal_trigger is None:
                terminal_trigger = "timeout"
                timed_out = True
            if (
                terminal_trigger == "cancellation"
                and group_exists
                and not term_attempted
            ):
                term_attempted = True
                term_sent = _signal_process_group(group_id, signal.SIGTERM)
                term_deadline = now + term_grace_seconds
            elif terminal_trigger == "timeout" and group_exists and not term_attempted:
                term_attempted = True
                term_sent = _signal_process_group(group_id, signal.SIGTERM)
                term_deadline = now + term_grace_seconds
            elif returncode is not None and group_exists and not term_attempted:
                term_attempted = True
                term_sent = _signal_process_group(group_id, signal.SIGTERM)
                term_deadline = now + term_grace_seconds
            if (
                term_attempted
                and group_exists
                and term_deadline is not None
                and now >= term_deadline
                and not kill_attempted
            ):
                kill_attempted = True
                kill_sent = _signal_process_group(group_id, _SIGKILL)
                kill_deadline = now + kill_grace_seconds

            wait_for = 0.02
            if term_deadline is not None and not kill_sent:
                wait_for = min(wait_for, max(0.0, term_deadline - now))
            events = selector.select(wait_for) if streams else ()
            for key, _mask in events:
                fd = key.fd
                name, stream = streams[fd]
                try:
                    chunk = os.read(fd, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(fd)
                    stream.close()
                    del streams[fd]
                elif name == "stdout":
                    stdout_truncated |= _append_bounded(stdout, chunk, max_output_bytes)
                else:
                    stderr_truncated |= _append_bounded(stderr, chunk, max_output_bytes)

            returncode = process.poll()
            group_exists = _process_group_exists(group_id)
            if returncode is not None and not group_exists:
                if not streams:
                    break
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + 0.1
                elif time.monotonic() >= drain_deadline:
                    for fd, (_name, stream) in list(streams.items()):
                        selector.unregister(fd)
                        stream.close()
                        del streams[fd]
                    break
            if kill_deadline is not None and time.monotonic() >= kill_deadline:
                if returncode is None:
                    try:
                        process.kill()
                        kill_sent = True
                    except OSError:
                        pass
                for fd, (_name, stream) in list(streams.items()):
                    selector.unregister(fd)
                    stream.close()
                    del streams[fd]
                break
    finally:
        selector.close()
        setattr(process, "_hermes_term_sent", term_sent)
        setattr(process, "_hermes_kill_sent", kill_sent)
    return (
        bytes(stdout),
        bytes(stderr),
        timed_out,
        stdout_truncated,
        stderr_truncated,
        cancellation_observed,
    )


def _cleanup_process_group(
    process: subprocess.Popen[bytes],
    *,
    term_grace_seconds: float,
    kill_grace_seconds: float,
) -> CleanupEvidence:
    pgid = process.pid
    term_sent = bool(getattr(process, "_hermes_term_sent", False))
    kill_sent = bool(getattr(process, "_hermes_kill_sent", False))
    if _process_group_exists(pgid) and not term_sent:
        term_sent = _signal_process_group(pgid, signal.SIGTERM)
        end = time.monotonic() + term_grace_seconds
        while _process_group_exists(pgid) and time.monotonic() < end:
            time.sleep(0.01)
    if _process_group_exists(pgid) and not kill_sent:
        kill_sent = _signal_process_group(pgid, _SIGKILL)
    try:
        process.wait(timeout=kill_grace_seconds)
        root_reaped = True
    except (subprocess.TimeoutExpired, ChildProcessError):
        try:
            process.kill()
            process.wait(timeout=kill_grace_seconds)
            root_reaped = True
            kill_sent = True
        except (OSError, subprocess.TimeoutExpired, ChildProcessError):
            root_reaped = process.poll() is not None
    end = time.monotonic() + kill_grace_seconds
    while _process_group_exists(pgid) and time.monotonic() < end:
        time.sleep(0.01)
    return CleanupEvidence(
        term_sent=term_sent,
        kill_sent=kill_sent,
        root_reaped=root_reaped,
        process_group_empty=not _process_group_exists(pgid),
    )


def _notify(callback: ReceiptCallbacks | None, method: str, **kwargs: Any) -> None:
    if callback is None:
        return
    getattr(callback, method)(**kwargs)


def _terminal_result(
    *,
    spec: ProcessRunSpec,
    confinement: str,
    state: str,
    root_pid: int | None,
    returncode: int | None,
    timed_out: bool,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    cleanup: CleanupEvidence | None = None,
    diagnostic: str | None = None,
) -> ProcessRunResult:
    exit_code = returncode if returncode is not None and returncode >= 0 else None
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    result = ProcessRunResult(
        backend=spec.backend,
        confinement=confinement,
        state=state,
        root_pid=root_pid,
        returncode=returncode,
        exit_code=exit_code,
        signal=signal_number,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        cleanup=cleanup or CleanupEvidence(requested=False),
        diagnostic=diagnostic,
    )
    _notify(spec.receipt, "on_terminal", state=state, result=result)
    return result


def _classify_terminal_state(
    *,
    returncode: int | None,
    timed_out: bool,
    broker_failed: bool,
    cleanup: CleanupEvidence,
    cancellation_requested: bool = False,
) -> tuple[str, str | None]:
    if cleanup.cgroup_empty is False:
        return "CONTAINMENT_FAILED", "dedicated cgroup was not empty after cleanup"
    if cleanup.root_reaped is False:
        return "FAILED", "root process was not reaped"
    if cleanup.process_group_empty is False:
        return "FAILED", "process group was not empty"
    if broker_failed:
        return "FAILED", "broker callback failed"
    if timed_out:
        return "TIMED_OUT", None
    if cancellation_requested:
        return "CANCELLED", None
    if returncode == 0 and cleanup.process_group_empty:
        return "SUCCEEDED", None
    return "FAILED", None


def run_owned_process(spec: ProcessRunSpec) -> ProcessRunResult:
    """Run one owned worker and return host-observed terminal evidence."""

    _validate_spec(spec)
    confinement = (
        PORTABLE_CONFINEMENT if spec.backend == "portable" else LINUX_STRICT_CONFINEMENT
    )
    try:
        _notify(
            spec.receipt,
            "on_created",
            backend=spec.backend,
            confinement=confinement,
        )
    except Exception as exc:
        return ProcessRunResult(
            backend=spec.backend,
            confinement=confinement,
            state=(
                "CONTAINMENT_FAILED" if spec.backend == "linux-strict" else "FAILED"
            ),
            root_pid=None,
            returncode=None,
            exit_code=None,
            signal=None,
            timed_out=False,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            cleanup=CleanupEvidence(requested=False),
            diagnostic=f"pre-spawn receipt callback failed: {type(exc).__name__}",
        )
    strict_probe: StrictPrerequisiteResult | None = None
    cgroup: Path | None = None
    cgroup_fd: int | None = None
    if spec.backend == "linux-strict":
        assert spec.strict is not None
        strict_probe = probe_linux_strict(spec.strict)
        if not strict_probe.available:
            return _terminal_result(
                spec=spec,
                confinement=confinement,
                state="CONTAINMENT_FAILED",
                root_pid=None,
                returncode=None,
                timed_out=False,
                diagnostic="; ".join(strict_probe.diagnostics),
            )

    environment = _minimal_environment(
        explicit=spec.environment,
        ambient_allowlist=spec.ambient_env_allowlist,
    )
    try:
        host_channel, child_channel = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
    except OSError as exc:
        state = "CONTAINMENT_FAILED" if spec.backend == "linux-strict" else "FAILED"
        return _terminal_result(
            spec=spec,
            confinement=confinement,
            state=state,
            root_pid=None,
            returncode=None,
            timed_out=False,
            diagnostic=f"capability socket creation failed: errno={exc.errno}",
        )
    child_fd = child_channel.fileno()
    pass_fds = [child_fd]
    if spec.backend == "portable":
        argv = [spec.executable, *spec.argv]
        if spec.capability_fd_arg is not None:
            argv.extend((spec.capability_fd_arg, str(child_fd)))
    else:
        assert strict_probe is not None and strict_probe.bwrap_path is not None
        try:
            bwrap_argv = _build_linux_strict_argv(
                spec, strict_probe.bwrap_path, capability_fd=child_fd
            )
        except (OSError, ValueError):
            host_channel.close()
            child_channel.close()
            raise
        assert spec.strict is not None
        try:
            cgroup, cgroup_fd = _create_owned_cgroup(spec.strict.cgroup_parent)
        except OSError as exc:
            host_channel.close()
            child_channel.close()
            return _terminal_result(
                spec=spec,
                confinement=confinement,
                state="CONTAINMENT_FAILED",
                root_pid=None,
                returncode=None,
                timed_out=False,
                diagnostic=f"dedicated cgroup creation failed: errno={exc.errno}",
            )
        worker_main = Path(__file__).with_name("subagent_worker_main.py")
        argv = [
            sys.executable,
            str(worker_main),
            "--enter-cgroup-fd",
            str(cgroup_fd),
            "--exec",
            *bwrap_argv,
        ]
        pass_fds.append(cgroup_fd)

    process: subprocess.Popen[bytes] | None = None
    stop_requested = threading.Event()
    broker_errors: list[str] = []
    broker_thread: threading.Thread | None = None
    broker_quiesced = False
    broker_active_operation: str | None = None
    broker_finalization_attempted = False

    def quiesce_broker() -> None:
        nonlocal broker_thread, broker_quiesced, broker_active_operation
        nonlocal broker_finalization_attempted
        if broker_finalization_attempted:
            return
        broker_finalization_attempted = True
        quiesced = True
        stop_requested.set()
        try:
            host_channel.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if broker_thread is not None:
            cancel_errors: list[BaseException] = []
            cancel = getattr(spec.broker, "cancel", None)
            if callable(cancel):
                cancel_result: list[Any] = []

                def revoke_broker() -> None:
                    try:
                        cancel_result.append(cancel())
                    except BaseException as exc:
                        cancel_errors.append(exc)

                revocation_thread = threading.Thread(
                    target=revoke_broker,
                    name=(
                        f"hermes-broker-revocation-"
                        f"{process.pid if process is not None else 'bootstrap'}"
                    ),
                    daemon=True,
                )
                revocation_thread.start()
                # Terminal evidence is immutable.  Once a side-effecting
                # operation is admitted, cancellation must finish before we
                # can publish that evidence; a deadline may trigger
                # cancellation, but it cannot authorize a premature result.
                revocation_thread.join(_BROKER_CONTAINMENT_FAIL_STOP_SECONDS)
                if revocation_thread.is_alive():
                    _fail_stop_broker_containment(
                        "broker cancellation callback exceeded containment deadline"
                    )
                    raise _BrokerContainmentFailStop(
                        "broker containment fail-stop returned"
                    )

                def admitted_operation() -> str | None:
                    try:
                        candidate = getattr(spec.broker, "active_operation_label", None)
                    except Exception:
                        candidate = None
                    if (
                        isinstance(candidate, str)
                        and 0 < len(candidate) <= 128
                        and all(
                            character.isascii()
                            and (character.isalnum() or character in "._:-")
                            for character in candidate
                        )
                    ):
                        return candidate
                    return None

                if cancel_errors:
                    quiesced = False
                    broker_active_operation = admitted_operation()
                    broker_errors.append(
                        f"broker revocation failed: {type(cancel_errors[0]).__name__}"
                    )
                elif cancel_result == [False]:
                    quiesced = False
                    active_operation = admitted_operation()
                    broker_active_operation = active_operation
                    operation_detail = (
                        f" {active_operation}" if active_operation is not None else ""
                    )
                    broker_errors.append(
                        "broker revocation deadline expired before admitted operation"
                        f"{operation_detail} quiesced"
                    )
            # Host-brokered operations may have side effects. Never finalize a
            # lifecycle result until every accepted operation has either
            # recorded its claim or failed; otherwise a late completion can
            # escape terminal classification and the result hash.
            broker_thread.join(_BROKER_CONTAINMENT_FAIL_STOP_SECONDS)
            if broker_thread.is_alive():
                _fail_stop_broker_containment(
                    "host broker operation exceeded containment deadline"
                )
                raise _BrokerContainmentFailStop(
                    "broker containment fail-stop returned"
                )
            broker_thread = None
            finalize = getattr(spec.broker, "finalize", None)
            if callable(finalize):
                finalize()
            quiesced = quiesced and not cancel_errors
        broker_quiesced = quiesced

    try:
        process = subprocess.Popen(
            argv,
            executable=argv[0],
            shell=False,
            cwd=str(spec.cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=tuple(pass_fds),
            start_new_session=True,
        )
        _notify(spec.receipt, "on_started", root_pid=process.pid)
        child_channel.close()
        if cgroup_fd is not None:
            os.close(cgroup_fd)
            cgroup_fd = None
        send_capability_secret(host_channel, spec.capability_secret)
        send_authenticated_frame(
            host_channel,
            {
                "capability_id": spec.capability_id,
                "launch_receipt_digest": spec.launch_receipt_digest,
            },
            spec.capability_secret,
        )

        if spec.broker is not None:
            broker = spec.broker

            def serve_broker() -> None:
                try:
                    broker.serve(
                        host_channel,
                        root_pid=process.pid,
                        stop_requested=stop_requested,
                    )
                except Exception as exc:
                    detail = _bounded_diagnostic_text(exc, limit=300)
                    broker_errors.append(
                        f"{type(exc).__name__}: {detail}"
                        if detail
                        else type(exc).__name__
                    )

            broker_thread = threading.Thread(
                target=serve_broker,
                name=f"hermes-broker-{process.pid}",
                daemon=True,
            )
            broker_thread.start()

        (
            stdout,
            stderr,
            timed_out,
            stdout_truncated,
            stderr_truncated,
            cancellation_observed,
        ) = _monitor_process(
            process,
            timeout_seconds=spec.timeout_seconds,
            term_grace_seconds=spec.term_grace_seconds,
            kill_grace_seconds=spec.kill_grace_seconds,
            max_output_bytes=spec.max_output_bytes,
            cancellation_event=spec.cancellation_event,
        )
        cleanup = _cleanup_process_group(
            process,
            term_grace_seconds=spec.term_grace_seconds,
            kill_grace_seconds=spec.kill_grace_seconds,
        )
        if cgroup is not None:
            cgroup_evidence = cleanup_cgroup(
                cgroup, timeout_seconds=spec.kill_grace_seconds
            )
            cleanup = dataclasses.replace(
                cleanup,
                descendant_scope="cgroup-v2",
                cgroup_kill_sent=cgroup_evidence.cgroup_kill_sent,
                cgroup_empty=cgroup_evidence.cgroup_empty,
                cgroup_removed=cgroup_evidence.cgroup_removed,
            )
        quiesce_broker()
        cleanup = dataclasses.replace(
            cleanup,
            broker_quiesced=broker_quiesced,
            broker_active_operation=broker_active_operation,
        )

        returncode = process.returncode
        state, diagnostic = _classify_terminal_state(
            returncode=returncode,
            timed_out=timed_out,
            broker_failed=bool(broker_errors),
            cleanup=cleanup,
            cancellation_requested=cancellation_observed,
        )
        if broker_errors and diagnostic == "broker callback failed":
            diagnostic = f"{diagnostic}: {broker_errors[0]}"
        return _terminal_result(
            spec=spec,
            confinement=confinement,
            state=state,
            root_pid=process.pid,
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            cleanup=cleanup,
            diagnostic=diagnostic,
        )
    except OSError as exc:
        cleanup = CleanupEvidence(requested=False)
        if process is not None:
            cleanup = _cleanup_process_group(
                process,
                term_grace_seconds=spec.term_grace_seconds,
                kill_grace_seconds=spec.kill_grace_seconds,
            )
        if cgroup is not None:
            cgroup_evidence = cleanup_cgroup(
                cgroup, timeout_seconds=spec.kill_grace_seconds
            )
            cleanup = dataclasses.replace(
                cleanup,
                descendant_scope="cgroup-v2",
                cgroup_kill_sent=cgroup_evidence.cgroup_kill_sent,
                cgroup_empty=cgroup_evidence.cgroup_empty,
                cgroup_removed=cgroup_evidence.cgroup_removed,
            )
        quiesce_broker()
        cleanup = dataclasses.replace(
            cleanup,
            broker_quiesced=broker_quiesced,
            broker_active_operation=broker_active_operation,
        )
        state = "CONTAINMENT_FAILED" if spec.backend == "linux-strict" else "FAILED"
        return _terminal_result(
            spec=spec,
            confinement=confinement,
            state=state,
            root_pid=process.pid if process is not None else None,
            returncode=process.returncode if process is not None else None,
            timed_out=False,
            cleanup=cleanup,
            diagnostic=f"owned process spawn/bootstrap failed: errno={exc.errno}",
        )
    except Exception as exc:
        cleanup = CleanupEvidence(requested=False)
        if process is not None:
            cleanup = _cleanup_process_group(
                process,
                term_grace_seconds=spec.term_grace_seconds,
                kill_grace_seconds=spec.kill_grace_seconds,
            )
        quiesce_broker()
        cleanup = dataclasses.replace(
            cleanup,
            broker_quiesced=broker_quiesced,
            broker_active_operation=broker_active_operation,
        )
        return ProcessRunResult(
            backend=spec.backend,
            confinement=confinement,
            state=(
                "CONTAINMENT_FAILED" if spec.backend == "linux-strict" else "FAILED"
            ),
            root_pid=process.pid if process is not None else None,
            returncode=process.returncode if process is not None else None,
            exit_code=(
                process.returncode
                if process is not None
                and process.returncode is not None
                and process.returncode >= 0
                else None
            ),
            signal=(
                -process.returncode
                if process is not None
                and process.returncode is not None
                and process.returncode < 0
                else None
            ),
            timed_out=False,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            cleanup=cleanup,
            diagnostic=f"post-spawn callback/bootstrap failed: {type(exc).__name__}",
        )
    finally:
        quiesce_broker()
        try:
            child_channel.close()
        except OSError:
            pass
        try:
            host_channel.close()
        except OSError:
            pass
        if cgroup_fd is not None:
            try:
                os.close(cgroup_fd)
            except OSError:
                pass
