"""Host-resolved execution profiles for delegated Hermes subagents.

Profiles are declared by the operator in ``config.yaml`` under
``delegation.execution_profiles.<profile_id>`` and resolved by the host at
launch time.  Plugin callers only ever supply a symbolic ``profile_id`` on
``SubagentLaunchRequest``; they never register or receive resolved profile
objects.

Security framing (deliberate, do not soften): ``in_process`` profiles pin
policy but provide no containment. ``portable`` profiles move the conversation
loop and exact local tools into an owned, scrubbed child process but remain
filesystem/network unconfined. ``linux_strict`` additionally requires a
host-delegated cgroup-v2 parent and Bubblewrap namespaces. The immutable launch
receipt records accepted policy; a separate execution receipt records observed
process/containment state. Remote named-agent attestation is out of scope.
"""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Optional

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_PROTOCOL_BYTES = 32_000  # Must fit the lifecycle context budget.
_MAX_TIMEOUT_SECONDS = 86_400.0
_MAX_PROCESS_ITERATIONS = 64
_VALID_EXECUTION_BACKENDS = frozenset({"in_process", "portable", "linux_strict"})
_VALID_ROLES = frozenset({"leaf", "orchestrator"})
UNSAFE_EXACT_PROFILE_TOOL_NAMES = frozenset(
    {
        "clarify",
        "delegate_task",
        "execute_code",
        "memory",
        "read_terminal",
        "session_search",
        "todo",
        "tool_search",
        "tool_describe",
        "tool_call",
    }
)
_PROFILE_KEYS = frozenset({
    "protocol_file",
    "role",
    "allowed_toolsets",
    "blocked_tools",
    "expected_tool_names",
    "allow_root",
    "allowed_child_profiles",
    "timeout_seconds",
    "execution_backend",
    "workspace_root",
    "cgroup_parent",
    "max_process_iterations",
})

# Honest, host-owned policy labels stamped into every profile launch receipt.
AMBIENT_POLICY_LABEL = (
    "ambient-shared: the child runs in the parent process and shares the "
    "parent's environment, filesystem, network, and credentials; this profile "
    "pins in-process policy only and provides no workspace, environment, or "
    "process confinement."
)
CLEANUP_POLICY_LABEL = (
    "cooperative-unconfirmed: cancellation and deadlines request interruption "
    "at the child's next safe boundary; termination and resource cleanup are "
    "not confirmed."
)
MCP_POLICY_LABEL = (
    "opted-out: profile launches never preserve ambient parent MCP toolsets; "
    "any MCP tools must be granted explicitly via allowed_toolsets and pinned "
    "by expected_tool_names."
)
DEADLINE_SEMANTICS_COOPERATIVE = (
    "cooperative: after timeout_seconds the host requests a hard interrupt "
    "and abandons the worker thread; the child is not forcibly terminated."
)
DEADLINE_SEMANTICS_NONE = (
    "none: no per-launch deadline; global delegation timeout config applies."
)

PORTABLE_AMBIENT_POLICY_LABEL = (
    "owned-process-scrubbed-unconfined: the conversation loop and exact local "
    "tools run in a separate process with a scrubbed environment; provider "
    "credentials stay in the parent, but filesystem and network are not OS-confined."
)
STRICT_AMBIENT_POLICY_LABEL = (
    "owned-process-linux-strict: the conversation loop and exact local tools run "
    "inside Bubblewrap filesystem/network namespaces and a host-delegated cgroup-v2; "
    "provider credentials stay in the parent."
)
PROCESS_CLEANUP_POLICY_LABEL = (
    "owned-confirmed: the execution receipt records process-group and, for Linux "
    "strict mode, cgroup cleanup evidence."
)
DEADLINE_SEMANTICS_HARD = (
    "hard-owned-process: timeout terminates and reaps the owned process group; "
    "Linux strict mode also kills and verifies the dedicated cgroup."
)


class ExecutionProfileError(ValueError):
    """A profile cannot be resolved or a profile launch must be refused."""


@dataclasses.dataclass(frozen=True)
class ResolvedExecutionProfile:
    """Immutable host-resolved profile snapshot.

    All fields are fixed at resolution time; the protocol file's bytes are
    read exactly once and pinned by ``protocol_sha256``.  Instances are
    host-only: they are attached to child agents as private attributes and
    are never returned across the plugin lifecycle API.
    """

    profile_id: str
    role: str
    allowed_toolsets: tuple[str, ...]
    expected_tool_names: frozenset[str]
    protocol_file: str
    protocol_text: str
    protocol_sha256: str
    allow_root: bool
    allowed_child_profiles: tuple[str, ...]
    timeout_seconds: Optional[float]
    execution_backend: str = "in_process"
    workspace_root: Optional[str] = None
    cgroup_parent: Optional[str] = None
    max_process_iterations: int = 8
    blocked_tools: frozenset[str] = frozenset()


@dataclasses.dataclass(frozen=True)
class SubagentLaunchReceipt:
    """Immutable, host-observed record of a profile's accepted launch state.

    Every field is derived from host state (the resolved profile, the
    constructed child agent, and the exact strings the host handed the
    child) — never from model output or plugin-supplied metadata. Provider
    and model are launch-time values; a later runtime failover is not
    reflected in this receipt. The
    ``ambient_policy`` / ``cleanup_policy`` labels are deliberately honest:
    environment and workspace are ambient/shared and cleanup is cooperative
    and unconfirmed.
    """

    receipt_version: int
    profile_id: str
    protocol_file: str
    protocol_sha256: str
    goal_sha256: str
    context_sha256: Optional[str]
    resolved_tool_names: tuple[str, ...]
    tool_schema_digest: str
    provider: Optional[str]
    model: Optional[str]
    role: str
    depth: int
    child_session_id: Optional[str]
    ambient_policy: str
    cleanup_policy: str
    mcp_policy: str
    deadline_seconds: Optional[float]
    deadline_semantics: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["resolved_tool_names"] = list(self.resolved_tool_names)
        return payload


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tool_schema_digest(tools: Any) -> str:
    """Deterministic digest of a child's exact tool schemas, sorted by name."""
    entries = []
    for tool in tools or []:
        if isinstance(tool, Mapping):
            fn = tool.get("function")
            entries.append(fn if isinstance(fn, Mapping) else tool)
    try:
        canonical = _json_dumps_sorted(
            sorted(entries, key=lambda e: str(e.get("name", "")))
        )
    except (TypeError, ValueError):
        canonical = repr(entries)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_dumps_sorted(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def validate_profile_id(profile_id: Any) -> str:
    if not isinstance(profile_id, str) or not _PROFILE_ID_RE.match(profile_id):
        raise ExecutionProfileError("profile_id must match ^[a-z0-9][a-z0-9_-]{0,63}$.")
    return profile_id


def _profiles_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    if not isinstance(config, Mapping):
        raise ExecutionProfileError(
            "Hermes config is unavailable; refusing profile launch."
        )
    try:
        config_snapshot = copy.deepcopy(dict(config))
    except Exception as exc:
        raise ExecutionProfileError(
            "Hermes config could not be snapshotted; refusing profile launch."
        ) from exc
    delegation = config_snapshot.get("delegation")
    profiles = (
        delegation.get("execution_profiles")
        if isinstance(delegation, Mapping)
        else None
    )
    if not isinstance(profiles, Mapping):
        return {}
    return profiles


def _known_toolset_names() -> set:
    """Toolsets recognized by the live registry (built-in + plugin + MCP)."""
    from toolsets import TOOLSETS

    known = set(TOOLSETS)
    try:
        from tools.registry import registry

        known.update(registry.get_available_toolsets())
    except Exception:
        # Fail closed to the static set — unknown plugin toolsets are then
        # rejected rather than silently accepted.
        pass
    return known


def _load_protocol(protocol_file: Any) -> tuple[str, str]:
    """Read the profile's protocol file once, fail-closed.

    Returns ``(text, sha256_hex)``.  Rejects absolute paths, parent traversal,
    every symlink path component, non-regular files, oversized files, and
    non-UTF-8 content.
    """
    if not isinstance(protocol_file, str) or not protocol_file.strip():
        raise ExecutionProfileError("protocol_file must be a non-empty relative path.")
    if "\x00" in protocol_file:
        raise ExecutionProfileError("protocol_file contains a NUL byte.")
    if (
        os.path.isabs(protocol_file)
        or protocol_file.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", protocol_file)
    ):
        raise ExecutionProfileError(
            "protocol_file must be relative to HERMES_HOME, not absolute."
        )
    # Validate raw string segments (Path() would normalize away "./"): no
    # empty, ".", or ".." components on either separator convention.
    segments = re.split(r"[/\\]", protocol_file)
    if not segments or any(seg in ("", ".", "..") for seg in segments):
        raise ExecutionProfileError(
            "protocol_file must not contain '.' or '..' segments."
        )

    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    try:
        home_real = home.resolve(strict=True)
    except OSError as exc:
        raise ExecutionProfileError(
            f"HERMES_HOME does not exist; cannot resolve protocol_file: {exc}"
        ) from exc
    root_fd: Optional[int] = None
    try:
        root_fd = _open_verified_root(home_real)
        literal_target = home_real.joinpath(*segments)
        data = _read_regular_file_beneath(
            home_real,
            literal_target,
            root_fd=root_fd,
        )
    finally:
        if root_fd is not None:
            os.close(root_fd)
    if len(data) > _MAX_PROTOCOL_BYTES:
        raise ExecutionProfileError(
            f"protocol_file exceeds {_MAX_PROTOCOL_BYTES} bytes."
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExecutionProfileError("protocol_file must be UTF-8 text.") from exc
    return text, hashlib.sha256(data).hexdigest()


def _open_verified_root(root: Path) -> int:
    """Open and identity-check the canonical Hermes root directory."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    fd: Optional[int] = None
    try:
        expected = os.stat(root, follow_symlinks=False)
        fd = os.open(root, os.O_RDONLY | directory | nofollow)
        actual = os.fstat(fd)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise ExecutionProfileError(
            "HERMES_HOME changed or became unsafe while being opened."
        ) from exc
    if (
        not stat.S_ISDIR(actual.st_mode)
        or (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino)
    ):
        os.close(fd)
        raise ExecutionProfileError(
            "HERMES_HOME identity changed while being opened."
        )
    return fd


def _read_regular_file_beneath(
    root: Path, target: Path, *, root_fd: Optional[int] = None
) -> bytes:
    """Open ``target`` beneath ``root`` without following path components.

    Descriptor-relative ``open`` calls ensure a directory or symlink swap
    between resolution and read cannot redirect the protocol outside the
    active Hermes home. Platforms without these primitives fail closed.
    """
    relative = target.relative_to(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    supports_dir_fd = os.open in getattr(os, "supports_dir_fd", set())
    if not (supports_dir_fd and nofollow and directory):
        raise ExecutionProfileError(
            "Execution profiles require descriptor-relative protocol traversal "
            "with O_NOFOLLOW and O_DIRECTORY; this platform is unsupported."
        )
    fd: Optional[int] = None
    try:
        fd = (
            os.dup(root_fd)
            if root_fd is not None
            else os.open(root, os.O_RDONLY | directory | nofollow)
        )
        for index, component in enumerate(relative.parts):
            flags = os.O_RDONLY | nofollow
            if index < len(relative.parts) - 1:
                flags |= directory
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ExecutionProfileError("protocol_file must be a regular file.")
        if st.st_size > _MAX_PROTOCOL_BYTES:
            raise ExecutionProfileError(
                f"protocol_file exceeds {_MAX_PROTOCOL_BYTES} bytes."
            )
        chunks: list[bytes] = []
        remaining = _MAX_PROTOCOL_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    except ExecutionProfileError:
        raise
    except OSError as exc:
        raise ExecutionProfileError(
            "protocol_file changed or became unsafe while being opened."
        ) from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ExecutionProfileError(f"{field} must be a non-empty list of strings.")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ExecutionProfileError(f"{field} entries must be non-empty strings.")
        out.append(item)
    if len(set(out)) != len(out):
        raise ExecutionProfileError(f"{field} must not contain duplicate entries.")
    return tuple(out)


def resolve_execution_profile(
    profile_id: str, *, config: Optional[Mapping[str, Any]] = None
) -> ResolvedExecutionProfile:
    """Resolve a symbolic profile id into an immutable host-owned profile.

    Every validation failure raises :class:`ExecutionProfileError`; there is
    no permissive fallback.  The protocol file is read exactly once here.
    """
    validate_profile_id(profile_id)
    profiles = _profiles_config(config)
    raw = profiles.get(profile_id)
    if not isinstance(raw, Mapping):
        raise ExecutionProfileError(
            f"Unknown execution profile {profile_id!r}: not declared under "
            "delegation.execution_profiles in config.yaml."
        )
    unknown_keys = set(raw) - _PROFILE_KEYS
    if unknown_keys:
        raise ExecutionProfileError(
            f"Unknown keys in profile {profile_id!r}: "
            f"{', '.join(sorted(str(key) for key in unknown_keys))}."
        )

    role = raw.get("role", "leaf")
    if role not in _VALID_ROLES:
        raise ExecutionProfileError("profile role must be 'leaf' or 'orchestrator'.")

    allowed_toolsets = _string_tuple(raw.get("allowed_toolsets"), "allowed_toolsets")
    unknown = set(allowed_toolsets) - _known_toolset_names()
    if unknown:
        raise ExecutionProfileError(
            f"Unknown toolsets in profile {profile_id!r}: {', '.join(sorted(unknown))}."
        )

    expected_tool_names = frozenset(
        _string_tuple(raw.get("expected_tool_names"), "expected_tool_names")
    )
    unsafe_names = expected_tool_names & UNSAFE_EXACT_PROFILE_TOOL_NAMES
    if unsafe_names:
        raise ExecutionProfileError(
            "expected_tool_names contains tools without frozen profile adapters: "
            f"{', '.join(sorted(unsafe_names))}."
        )

    raw_blocked = raw.get("blocked_tools", ())
    if raw_blocked in (None, (), []):
        blocked_tools: frozenset[str] = frozenset()
    else:
        blocked_tools = frozenset(_string_tuple(raw_blocked, "blocked_tools"))
    overlap = blocked_tools & expected_tool_names
    if overlap:
        raise ExecutionProfileError(
            "blocked_tools and expected_tool_names must be disjoint: "
            f"{', '.join(sorted(overlap))}."
        )

    protocol_file = raw.get("protocol_file")
    protocol_text, protocol_sha = _load_protocol(protocol_file)

    allow_root = raw.get("allow_root", False)
    if not isinstance(allow_root, bool):
        raise ExecutionProfileError("allow_root must be a boolean.")

    raw_children = raw.get("allowed_child_profiles", ())
    if raw_children in (None, (), []):
        allowed_child_profiles: tuple[str, ...] = ()
    else:
        allowed_child_profiles = tuple(
            validate_profile_id(child)
            for child in _string_tuple(raw_children, "allowed_child_profiles")
        )

    timeout_seconds = raw.get("timeout_seconds")
    if timeout_seconds is not None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS
        ):
            raise ExecutionProfileError(
                f"timeout_seconds must be a number in (0, {int(_MAX_TIMEOUT_SECONDS)}]."
            )
        timeout_seconds = float(timeout_seconds)

    execution_backend = raw.get("execution_backend", "in_process")
    if execution_backend not in _VALID_EXECUTION_BACKENDS:
        raise ExecutionProfileError(
            "execution_backend must be 'in_process', 'portable', or 'linux_strict'."
        )
    workspace_root = raw.get("workspace_root")
    cgroup_parent = raw.get("cgroup_parent")
    if execution_backend == "in_process":
        if workspace_root is not None or cgroup_parent is not None:
            raise ExecutionProfileError(
                "workspace_root/cgroup_parent are only valid for process execution backends."
            )
    else:
        if not isinstance(workspace_root, str) or not os.path.isabs(workspace_root):
            raise ExecutionProfileError(
                "process execution backends require an absolute workspace_root."
            )
        try:
            resolved_workspace = Path(workspace_root).resolve(strict=True)
        except OSError as exc:
            raise ExecutionProfileError(
                "workspace_root must be an existing directory."
            ) from exc
        if not resolved_workspace.is_dir() or resolved_workspace == Path(resolved_workspace.anchor):
            raise ExecutionProfileError(
                "workspace_root must be an existing non-root directory."
            )
        workspace_root = str(resolved_workspace)
    if execution_backend == "linux_strict":
        if not isinstance(cgroup_parent, str) or not os.path.isabs(cgroup_parent):
            raise ExecutionProfileError(
                "linux_strict execution requires an absolute host-delegated cgroup_parent."
            )
        try:
            resolved_cgroup_parent = Path(cgroup_parent).resolve(strict=True)
        except OSError as exc:
            raise ExecutionProfileError(
                "cgroup_parent must be an existing directory."
            ) from exc
        if (
            not resolved_cgroup_parent.is_dir()
            or resolved_cgroup_parent == Path(resolved_cgroup_parent.anchor)
        ):
            raise ExecutionProfileError(
                "cgroup_parent must be an existing non-root directory."
            )
        cgroup_parent = str(resolved_cgroup_parent)
    elif cgroup_parent is not None:
        raise ExecutionProfileError(
            "cgroup_parent is only valid for linux_strict execution."
        )
    max_process_iterations = raw.get("max_process_iterations", 8)
    if (
        isinstance(max_process_iterations, bool)
        or not isinstance(max_process_iterations, int)
        or not 1 <= max_process_iterations <= _MAX_PROCESS_ITERATIONS
    ):
        raise ExecutionProfileError(
            f"max_process_iterations must be an integer in [1, {_MAX_PROCESS_ITERATIONS}]."
        )

    return ResolvedExecutionProfile(
        profile_id=profile_id,
        role=role,
        allowed_toolsets=allowed_toolsets,
        expected_tool_names=expected_tool_names,
        protocol_file=protocol_file,
        protocol_text=protocol_text,
        protocol_sha256=protocol_sha,
        allow_root=allow_root,
        allowed_child_profiles=allowed_child_profiles,
        timeout_seconds=timeout_seconds,
        execution_backend=execution_backend,
        workspace_root=workspace_root,
        cgroup_parent=cgroup_parent,
        max_process_iterations=max_process_iterations,
        blocked_tools=blocked_tools,
    )


def check_profile_transition(
    parent_agent: Any, profile: ResolvedExecutionProfile
) -> None:
    """Enforce the host-observed profile transition graph.

    The decision key is the parent's host-set ``_execution_profile_id`` /
    ``_execution_profile`` attributes (stamped by the launch path), never
    model-visible metadata.  A parent with no profile is a root launch and
    requires ``allow_root: true`` on the target profile; a profile parent may
    only launch profiles listed in its own ``allowed_child_profiles``.
    """
    parent_profile_id = getattr(parent_agent, "_execution_profile_id", None)
    if parent_profile_id is None:
        if not profile.allow_root:
            raise ExecutionProfileError(
                f"Profile {profile.profile_id!r} does not allow root launches "
                "(allow_root is false)."
            )
        return
    parent_profile = getattr(parent_agent, "_execution_profile", None)
    if not isinstance(parent_profile, ResolvedExecutionProfile) or (
        parent_profile.profile_id != parent_profile_id
    ):
        # A profile id without its pinned resolved profile is an integrity
        # failure — refuse rather than re-resolving from mutable config.
        raise ExecutionProfileError(
            "Parent profile state is inconsistent; refusing nested profile launch."
        )
    if profile.profile_id not in parent_profile.allowed_child_profiles:
        raise ExecutionProfileError(
            f"Profile {parent_profile_id!r} may not launch profile "
            f"{profile.profile_id!r}: not in its allowed_child_profiles."
        )
