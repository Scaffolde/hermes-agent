"""Public, plugin-safe lifecycle API for delegated Hermes subagents.

This module deliberately exposes immutable contracts, not ``AIAgent`` objects.
It is the supported boundary for plugins that need to supervise fresh child
sessions; plugins must obtain it from ``PluginContext.subagent_lifecycle``.
"""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent.interrupt_compat import request_hard_interrupt
from agent.subagent_execution_profiles import (
    AMBIENT_POLICY_LABEL,
    CLEANUP_POLICY_LABEL,
    DEADLINE_SEMANTICS_COOPERATIVE,
    DEADLINE_SEMANTICS_HARD,
    DEADLINE_SEMANTICS_NONE,
    MCP_POLICY_LABEL,
    PORTABLE_AMBIENT_POLICY_LABEL,
    PROCESS_CLEANUP_POLICY_LABEL,
    STRICT_AMBIENT_POLICY_LABEL,
    ExecutionProfileError,
    ResolvedExecutionProfile,
    SubagentLaunchReceipt,
    UNSAFE_EXACT_PROFILE_TOOL_NAMES,
    check_profile_transition,
    resolve_execution_profile,
    sha256_text,
    tool_schema_digest,
)
from agent.subagent_execution_receipts import (
    SubagentExecutionReceipt,
    SubagentExecutionRecorder,
)

PUBLIC_CONTRACT_VERSION = 1
_MAX_GOAL_CHARS = 16_000
_MAX_CONTEXT_CHARS = 32_000
_MAX_METADATA_BYTES = 8_192
_MAX_RESULT_CHARS = 32_000
_TERMINAL_RETENTION_SECONDS = 3_600


class SubagentLifecycleError(ValueError):
    """A request cannot be safely accepted by the public lifecycle API."""


class SubagentState(str, enum.Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class SubagentLaunchRequest:
    goal: str
    context: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    # Symbolic id of a host-declared execution profile
    # (config.yaml: delegation.execution_profiles.<id>).  With the default
    # None, launch behavior is byte-for-byte the legacy path.
    profile_id: Optional[str] = None
    # Optional caller claim that must match the host-resolved profile workspace.
    # This does not select a workspace; it binds typed plugin inputs to the
    # authority the host already declared in the named execution profile.
    required_workspace_root: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentHandle:
    contract_version: int
    subagent_id: str
    parent_session_id: Optional[str]
    correlation_id: Optional[str]
    created_at: float
    provider: Optional[str]
    model: Optional[str]
    role: str
    depth: int
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("Malformed subagent handle.") from exc


@dataclasses.dataclass(frozen=True)
class SubagentStatus:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentTerminalState:
    handle: SubagentHandle
    state: SubagentState
    completed: bool
    timed_out: bool = False
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentCancelResult:
    accepted: bool
    already_terminal: bool = False
    unknown_handle: bool = False
    unsupported: bool = False
    state: SubagentState = SubagentState.UNKNOWN


@dataclasses.dataclass(frozen=True)
class SubagentResult:
    handle: SubagentHandle
    terminal_state: SubagentState
    ready: bool
    summary: Optional[str] = None
    structured_payload: Optional[Mapping[str, Any]] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_classification: Optional[str] = None
    error_message: Optional[str] = None
    usage_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    tool_execution_summary: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    result_hash: Optional[str] = None
    # Host-owned launch receipt (profile launches only; None for legacy
    # launches, and then excluded from the result hash so legacy hashes are
    # unchanged).  Frozen dataclass keeps the public snapshot immutable;
    # dataclasses.asdict still makes result hashing/serialization deterministic.
    launch_receipt: Optional[SubagentLaunchReceipt] = None
    execution_receipt: Optional[SubagentExecutionReceipt] = None
    execution_receipt_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentReconnectResult:
    connected: bool
    state: SubagentState
    diagnostic: Optional[str] = None


@dataclasses.dataclass
class _Record:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    agent: Any = None
    future: Optional[Future] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[SubagentResult] = None
    receipt: Optional[SubagentLaunchReceipt] = None
    execution_profile: Optional[ResolvedExecutionProfile] = None
    execution_receipt: Optional[SubagentExecutionReceipt] = None
    timeout_override: Optional[float] = None
    start_gate: Optional[threading.Event] = None
    launch_committed: bool = True


class _Registry:
    """Thread-safe terminal-retention registry; never returns live records."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.records: dict[str, _Record] = {}
        self.correlations: dict[tuple[Optional[str], str], str] = {}
        self.pending_correlations: set[tuple[Optional[str], str]] = set()
        self.pending_profile_launches = 0


_REGISTRY = _Registry()
# Daemon worker pool: a wedged/abandoned child must never block interpreter
# exit at atexit-join time (same rationale as _run_single_child's timeout
# executor and the async-delegation registry pool).
from tools.daemon_pool import DaemonThreadPoolExecutor as _DaemonExecutor

_EXECUTOR_MAX_WORKERS = 8
_EXECUTOR = _DaemonExecutor(
    max_workers=_EXECUTOR_MAX_WORKERS, thread_name_prefix="hermes-lifecycle"
)
_SECRET = secrets.token_bytes(32)
_ACTIVE_PARENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "hermes_subagent_lifecycle_parent", default=None
)


@contextmanager
def bind_subagent_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn."""
    token = _ACTIVE_PARENT_AGENT.set(parent_agent)
    try:
        yield
    finally:
        _ACTIVE_PARENT_AGENT.reset(token)


def get_active_subagent_parent() -> Any:
    """Return the parent bound to this execution context, if any."""
    return _ACTIVE_PARENT_AGENT.get()


class SubagentLifecycleService:
    """Stable public service returned by :attr:`PluginContext.subagent_lifecycle`.

    Running children are in-process only.  Completed results remain available
    until process exit; ``reconnect`` accurately reports that a serialized
    handle cannot reconnect after a restart instead of launching work again.
    """

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def launch(self, request: SubagentLaunchRequest) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError(
                "No active Hermes parent session is available."
            )
        self._validate_request(request, parent)
        profile = self._resolve_profile_for_launch(request, parent)
        self._validate_profile_workspace_binding(request, profile)
        parent_session_id = str(getattr(parent, "session_id", "") or "") or None
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError(
                "parent_session_id does not match the active session."
            )
        correlation_key = (parent_session_id, request.correlation_id or "")
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and (
                correlation_key in _REGISTRY.correlations
                or correlation_key in _REGISTRY.pending_correlations
            ):
                raise SubagentLifecycleError(
                    "Duplicate correlation_id for this parent session."
                )
            if profile is not None:
                # Fail fast before executor saturation: nested profile
                # launches (candidate -> verifier) deadlock if every worker
                # is occupied by a parent waiting on a queued child.  Legacy
                # launches keep their historical silent-queue behavior.
                inflight = (
                    sum(1 for r in _REGISTRY.records.values() if r.result is None)
                    + _REGISTRY.pending_profile_launches
                )
                if inflight >= _EXECUTOR_MAX_WORKERS:
                    raise SubagentLifecycleError(
                        f"Subagent executor is saturated ({inflight} in-flight "
                        f">= {_EXECUTOR_MAX_WORKERS} workers); profile launch "
                        "fails fast instead of queueing to avoid nested-"
                        "delegation deadlock."
                    )
                _REGISTRY.pending_profile_launches += 1
            if request.correlation_id:
                _REGISTRY.pending_correlations.add(correlation_key)

        profile_slot_reserved = profile is not None
        correlation_reserved = request.correlation_id is not None
        child: Any = None
        try:
            launch_context = request.context
            if profile is not None:
                combined_size = len(profile.protocol_text) + (
                    len(request.context) + 2 if request.context is not None else 0
                )
                if combined_size > _MAX_CONTEXT_CHARS:
                    raise SubagentLifecycleError(
                        "Profile protocol plus request context exceeds 32000 characters."
                    )

            # Delegate construction remains internal so plugin code never imports
            # private delegation helpers or manipulates the active-child registry.
            from tools.delegate_tool import (
                _build_child_preserving_parent_tools,
                DEFAULT_MAX_ITERATIONS,
            )

            build_kwargs: dict[str, Any] = {}
            if profile is not None:
                # Explicit per-launch opt-out of ambient MCP preservation; the
                # legacy path passes nothing so global config keeps governing it.
                build_kwargs["inherit_mcp"] = False
                build_kwargs["exact_tool_catalog"] = True
                build_kwargs["system_prompt_override"] = profile.protocol_text
            child = _build_child_preserving_parent_tools(
                task_index=0,
                goal=request.goal,
                context=launch_context,
                toolsets=list(profile.allowed_toolsets)
                if profile is not None
                else (
                    list(request.allowed_toolsets) if request.allowed_toolsets else None
                ),
                model=request.model,
                max_iterations=DEFAULT_MAX_ITERATIONS,
                task_count=1,
                parent_agent=parent,
                role=profile.role if profile is not None else request.role,
                **build_kwargs,
            )
            subagent_id = str(getattr(child, "_subagent_id", "") or "")
            if not subagent_id:
                self._abort_child_launch(child, parent)
                child = None
                raise SubagentLifecycleError(
                    "Hermes failed to assign a child identity."
                )
            created = time.time()
            receipt: Optional[SubagentLaunchReceipt] = None
            if profile is not None:
                try:
                    receipt = self._enforce_profile_contract(
                        child, profile, request, launch_context, created
                    )
                except BaseException:
                    self._abort_child_launch(child, parent)
                    child = None
                    raise
                if getattr(child, "api_mode", None) == "codex_responses" and (
                    getattr(child, "provider", None) in {"xai", "xai-oauth"}
                    or getattr(child, "_base_url_hostname", None) == "api.x.ai"
                ):
                    self._abort_child_launch(child, parent)
                    child = None
                    raise SubagentLifecycleError(
                        "Exact execution profiles do not support xAI Responses because "
                        "its provider wire path rewrites tool schemas after launch binding."
                    )
                if (
                    profile.execution_backend != "in_process"
                    and getattr(child, "api_mode", None) != "chat_completions"
                ):
                    self._abort_child_launch(child, parent)
                    child = None
                    raise SubagentLifecycleError(
                        "Process execution profiles support only api_mode "
                        "'chat_completions'; refusing before process spawn."
                    )
            handle = SubagentHandle(
                PUBLIC_CONTRACT_VERSION,
                subagent_id,
                parent_session_id,
                request.correlation_id,
                created,
                getattr(child, "provider", None),
                getattr(child, "model", None),
                getattr(child, "_delegate_role", request.role),
                int(getattr(child, "_delegate_depth", 1) or 1),
                self._capability(subagent_id, parent_session_id, created),
            )
            record = _Record(
                handle,
                SubagentState.PENDING,
                created,
                agent=child,
                receipt=receipt,
                execution_profile=profile,
                timeout_override=(
                    profile.timeout_seconds if profile is not None else None
                ),
                start_gate=threading.Event() if profile is not None else None,
                launch_committed=profile is None,
            )
            with _REGISTRY.lock:
                _REGISTRY.records[subagent_id] = record
                if request.correlation_id:
                    _REGISTRY.pending_correlations.discard(correlation_key)
                    _REGISTRY.correlations[correlation_key] = subagent_id
                    correlation_reserved = False
                if profile_slot_reserved:
                    _REGISTRY.pending_profile_launches -= 1
                    profile_slot_reserved = False
            execution_goal = request.goal
            if profile is not None and request.context is not None:
                execution_goal = f"{request.goal}\n\nContext:\n{request.context}"
            try:
                record.future = _EXECUTOR.submit(
                    self._run, record, execution_goal, parent
                )
            except BaseException:
                with _REGISTRY.lock:
                    _REGISTRY.records.pop(subagent_id, None)
                    if request.correlation_id:
                        _REGISTRY.correlations.pop(correlation_key, None)
                self._abort_child_launch(child, parent)
                child = None
                raise
            if profile is not None:
                try:
                    for attr in (
                        "_publish_deferred_context_session_start",
                        "_publish_deferred_launch",
                    ):
                        publish = getattr(child, attr, None)
                        if callable(publish):
                            publish()
                            delattr(child, attr)
                    record.launch_committed = True
                except BaseException:
                    with _REGISTRY.lock:
                        _REGISTRY.records.pop(subagent_id, None)
                        if request.correlation_id:
                            _REGISTRY.correlations.pop(correlation_key, None)
                    self._abort_child_launch(child, parent)
                    child = None
                    raise
                finally:
                    if record.start_gate is not None:
                        record.start_gate.set()
            return handle
        finally:
            if profile_slot_reserved or correlation_reserved:
                with _REGISTRY.lock:
                    if profile_slot_reserved:
                        _REGISTRY.pending_profile_launches -= 1
                    if correlation_reserved:
                        _REGISTRY.pending_correlations.discard(correlation_key)

    def describe(self, handle: SubagentHandle) -> Optional[SubagentLaunchReceipt]:
        """Return the immutable host-owned launch receipt, pre-completion.

        Only profile launches carry receipts; legacy launches and unknown,
        forged, or foreign handles return ``None``.
        """
        record = self._record(handle)
        return record.receipt if record is not None else None

    def describe_execution(
        self, handle: SubagentHandle
    ) -> Optional[SubagentExecutionReceipt]:
        """Return the separate immutable process execution receipt."""
        record = self._record(handle)
        return record.execution_receipt if record is not None else None

    @staticmethod
    def _resolve_profile_for_launch(
        request: SubagentLaunchRequest, parent: Any
    ) -> Optional[ResolvedExecutionProfile]:
        """Host-resolve and gate the requested execution profile (fail-closed)."""
        if request.profile_id is None:
            if getattr(parent, "_execution_profile_id", None) is not None:
                raise SubagentLifecycleError(
                    "A profiled parent may only launch a named execution profile."
                )
            return None
        if request.allowed_toolsets is not None:
            raise SubagentLifecycleError(
                "allowed_toolsets is profile-owned; a profile launch must not "
                "pass its own toolsets."
            )
        if request.role != "leaf":
            raise SubagentLifecycleError(
                "role is profile-owned; a profile launch must not request a role."
            )
        try:
            profile = resolve_execution_profile(request.profile_id)
            check_profile_transition(parent, profile)
        except ExecutionProfileError as exc:
            raise SubagentLifecycleError(str(exc)) from exc
        return profile

    @staticmethod
    def _validate_profile_workspace_binding(
        request: SubagentLaunchRequest,
        profile: Optional[ResolvedExecutionProfile],
    ) -> None:
        claimed_root = request.required_workspace_root
        if claimed_root is None:
            return
        if profile is None or profile.workspace_root is None:
            raise SubagentLifecycleError(
                "required_workspace_root requires a named profile with a host-owned workspace_root."
            )
        claimed = Path(claimed_root).expanduser()
        if not claimed.is_absolute():
            raise SubagentLifecycleError(
                "required_workspace_root must be an absolute path."
            )
        if claimed.resolve() != Path(profile.workspace_root).expanduser().resolve():
            raise SubagentLifecycleError(
                "required_workspace_root does not match the host-owned execution profile workspace."
            )

    @staticmethod
    def _enforce_profile_contract(
        child: Any,
        profile: ResolvedExecutionProfile,
        request: SubagentLaunchRequest,
        launch_context: Optional[str],
        created: float,
    ) -> SubagentLaunchReceipt:
        """Post-construction strict checks + host-observed receipt.

        Role must not have degraded and the child's post-``check_fn`` tool
        surface must equal the profile's expected set exactly. Profile authors
        must list every effective tool explicitly or the launch fails and the
        child is closed.
        """
        effective_role_value = getattr(child, "_delegate_role", None)
        if effective_role_value != profile.role:
            raise SubagentLifecycleError(
                f"Profile {profile.profile_id!r} requires role "
                f"{profile.role!r} but the child resolved to "
                f"{effective_role_value!r} (depth or kill-switch bound); profile "
                "launches never silently degrade."
            )
        effective_role = str(effective_role_value)
        from tools.registry import registry
        from tools.mcp_tool import _agent_tools_lock

        # One boundary covers the live schema read, dynamic-route rejection,
        # registry handler capture, and freeze publication. MCP refresh uses the
        # same lock for its final publish, so it cannot pass a stale freeze
        # check and broaden the child after this contract is accepted.
        with _agent_tools_lock:
            blocked = set(profile.blocked_tools)
            if blocked:
                valid_before_block = getattr(child, "valid_tool_names", None)
                if not isinstance(valid_before_block, (set, frozenset)):
                    raise SubagentLifecycleError(
                        f"Profile {profile.profile_id!r} cannot apply blocked_tools "
                        "because the child tool catalog is unavailable."
                    )
                child.valid_tool_names = set(valid_before_block) - blocked
                filtered_tools = []
                for tool in getattr(child, "tools", None) or []:
                    function = tool.get("function") if isinstance(tool, dict) else None
                    name = function.get("name") if isinstance(function, dict) else None
                    if name not in blocked:
                        filtered_tools.append(tool)
                child.tools = filtered_tools

            valid = getattr(child, "valid_tool_names", None)
            actual = frozenset(valid) if isinstance(valid, (set, frozenset)) else None
            if actual is None or actual != profile.expected_tool_names:
                missing = sorted(profile.expected_tool_names - (actual or frozenset()))
                unexpected = sorted(
                    (actual or frozenset()) - profile.expected_tool_names
                )
                raise SubagentLifecycleError(
                    f"Profile {profile.profile_id!r} exact tool contract violated: "
                    f"missing={missing} unexpected={unexpected}."
                )

            effective_schemas = {}
            for tool in getattr(child, "tools", None) or []:
                function = tool.get("function") if isinstance(tool, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(name, str):
                    effective_schemas[name] = function
            schema_names = set(effective_schemas)
            if schema_names != set(actual):
                missing = sorted(set(actual) - schema_names)
                unexpected = sorted(schema_names - set(actual))
                raise SubagentLifecycleError(
                    f"Profile {profile.profile_id!r} exact schema contract violated: "
                    f"missing={missing} unexpected={unexpected}."
                )

            unsafe = set(actual & UNSAFE_EXACT_PROFILE_TOOL_NAMES)
            unsafe.update(
                actual
                & set(getattr(child, "_context_engine_tool_names", set()) or set())
            )
            memory_manager = getattr(child, "_memory_manager", None)
            if memory_manager is not None:
                try:
                    unsafe.update(
                        name for name in actual if memory_manager.has_tool(name)
                    )
                except Exception as exc:
                    raise SubagentLifecycleError(
                        "Could not verify external-memory tool routing for exact profile."
                    ) from exc
            if unsafe:
                raise SubagentLifecycleError(
                    "Exact profile contains tools without frozen dispatch adapters: "
                    f"{sorted(unsafe)}."
                )

            built_generation = getattr(
                child, "_delegate_tool_registry_generation", None
            )
            snapshot_generation, dispatch_entries = (
                registry.snapshot_dispatch_entries_with_generation(
                    set(actual),
                    effective_schemas=effective_schemas,
                )
            )
            if built_generation != snapshot_generation:
                raise SubagentLifecycleError(
                    "Tool registry changed while the exact profile child was being built; "
                    "refusing a mixed schema/handler snapshot."
                )
            if set(dispatch_entries) != set(actual):
                missing_handlers = sorted(set(actual) - set(dispatch_entries))
                raise SubagentLifecycleError(
                    f"Exact profile tool handlers are unavailable for: {missing_handlers}."
                )
            # Host-only attrs: nested transition checks read these off the active
            # parent; the MCP-refresh freeze guard pins the tool surface so later
            # registry/MCP refreshes can never broaden a strict child.
            child._execution_profile_id = profile.profile_id
            child._execution_profile = profile
            child._delegate_frozen_tool_names = frozenset(profile.expected_tool_names)
            child._delegate_frozen_dispatch_entries = dispatch_entries
            schema_digest = tool_schema_digest(getattr(child, "tools", None))
        return SubagentLaunchReceipt(
            receipt_version=1,
            profile_id=profile.profile_id,
            protocol_file=profile.protocol_file,
            protocol_sha256=profile.protocol_sha256,
            goal_sha256=sha256_text(request.goal),
            context_sha256=(
                sha256_text(launch_context) if launch_context is not None else None
            ),
            resolved_tool_names=tuple(sorted(actual)),
            tool_schema_digest=schema_digest,
            provider=getattr(child, "provider", None),
            model=getattr(child, "model", None),
            role=effective_role,
            depth=int(getattr(child, "_delegate_depth", 1) or 1),
            child_session_id=(str(getattr(child, "session_id", "") or "") or None),
            ambient_policy=(
                AMBIENT_POLICY_LABEL
                if profile.execution_backend == "in_process"
                else (
                    PORTABLE_AMBIENT_POLICY_LABEL
                    if profile.execution_backend == "portable"
                    else STRICT_AMBIENT_POLICY_LABEL
                )
            ),
            cleanup_policy=(
                CLEANUP_POLICY_LABEL
                if profile.execution_backend == "in_process"
                else PROCESS_CLEANUP_POLICY_LABEL
            ),
            mcp_policy=MCP_POLICY_LABEL,
            deadline_seconds=profile.timeout_seconds,
            deadline_semantics=(
                DEADLINE_SEMANTICS_NONE
                if profile.timeout_seconds is None
                else (
                    DEADLINE_SEMANTICS_COOPERATIVE
                    if profile.execution_backend == "in_process"
                    else DEADLINE_SEMANTICS_HARD
                )
            ),
            created_at=created,
        )

    @staticmethod
    def _abort_child_launch(child: Any, parent: Any) -> None:
        """Best-effort teardown of a child whose launch was refused.

        Cleanup here is cooperative and unconfirmed by design: the child never
        ran, but its constructor may have opened tool resources.
        """
        try:
            if hasattr(parent, "_active_children"):
                lock = getattr(parent, "_active_children_lock", None)
                if lock:
                    with lock:
                        if child in parent._active_children:
                            parent._active_children.remove(child)
                elif child in parent._active_children:
                    parent._active_children.remove(child)
        except Exception:
            pass
        try:
            if hasattr(child, "close"):
                child.close()
        except Exception:
            pass

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(
                handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE"
            )
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def wait(
        self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None
    ) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            return SubagentTerminalState(
                handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE"
            )
        future = record.future
        if future is not None:
            try:
                future.result(timeout=timeout_seconds)
            except TimeoutError:
                return SubagentTerminalState(record.handle, record.state, False, True)
            except Exception:
                pass
        with _REGISTRY.lock:
            return SubagentTerminalState(
                record.handle, record.state, record.result is not None
            )

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        record = self._record(handle)
        if record is None:
            return SubagentCancelResult(False, unknown_handle=True)
        with _REGISTRY.lock:
            if record.result is not None:
                return SubagentCancelResult(
                    False, already_terminal=True, state=record.state
                )
            agent = record.agent
            record.state = SubagentState.CANCEL_REQUESTED
            record.updated_at = time.time()
        if agent is None:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        try:
            accepted = request_hard_interrupt(
                agent,
                f"Lifecycle cancellation requested: {reason[:500]}",
                tool_reason="subagent cancellation requested",
            )
        except Exception:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        if not accepted:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        return SubagentCancelResult(True, state=SubagentState.CANCEL_REQUESTED)

    def result(self, handle: SubagentHandle) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(
                handle,
                SubagentState.UNKNOWN,
                False,
                error_classification="UNKNOWN_HANDLE",
            )
        with _REGISTRY.lock:
            if record.result is not None:
                return record.result
            return SubagentResult(
                record.handle, record.state, False, error_classification="NOT_READY"
            )

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(
                False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE"
            )
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        if (
            not isinstance(handle, SubagentHandle)
            or type(handle.contract_version) is not int
            or handle.contract_version != PUBLIC_CONTRACT_VERSION
        ):
            return None
        if (
            not isinstance(handle.subagent_id, str)
            or not handle.subagent_id
            or (
                handle.parent_session_id is not None
                and not isinstance(handle.parent_session_id, str)
            )
            or (
                handle.correlation_id is not None
                and not isinstance(handle.correlation_id, str)
            )
            or isinstance(handle.created_at, bool)
            or not isinstance(handle.created_at, (int, float))
            or not math.isfinite(handle.created_at)
            or (handle.provider is not None and not isinstance(handle.provider, str))
            or (handle.model is not None and not isinstance(handle.model, str))
            or not isinstance(handle.role, str)
            or type(handle.depth) is not int
            or not isinstance(handle.capability, str)
        ):
            return None
        if not hmac.compare_digest(
            handle.capability,
            self._capability(
                handle.subagent_id, handle.parent_session_id, handle.created_at
            ),
        ):
            return None
        parent = self._parent_agent_resolver()
        active_parent_id = str(getattr(parent, "session_id", "") or "") or None
        if active_parent_id != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            return _REGISTRY.records.get(handle.subagent_id)

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            subagent_id
            for subagent_id, record in _REGISTRY.records.items()
            if record.result is not None
            and record.completed_at is not None
            and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            record = _REGISTRY.records.pop(subagent_id)
            if record.handle.correlation_id:
                _REGISTRY.correlations.pop(
                    (record.handle.parent_session_id, record.handle.correlation_id),
                    None,
                )

    def _run(self, record: _Record, goal: str, parent: Any) -> None:
        if record.start_gate is not None:
            record.start_gate.wait()
            if not record.launch_committed:
                return
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = time.time()
            record.updated_at = record.started_at
        receipt_payload = record.receipt
        try:
            profile = record.execution_profile
            if profile is not None and profile.execution_backend != "in_process":
                raw = self._run_process_child(record, goal, profile)
            else:
                from tools.delegate_tool import _run_child_lifecycle

                if record.timeout_override is not None:
                    raw = _run_child_lifecycle(
                        0,
                        goal,
                        record.agent,
                        parent,
                        timeout_override=record.timeout_override,
                    )
                else:
                    raw = _run_child_lifecycle(0, goal, record.agent, parent)
            status = (
                str(raw.get("status", "error")) if isinstance(raw, dict) else "error"
            )
            if status == "completed":
                state = SubagentState.SUCCEEDED
            elif status == "interrupted":
                state = (
                    SubagentState.CANCELLED
                    if record.state == SubagentState.CANCEL_REQUESTED
                    else SubagentState.INTERRUPTED
                )
            else:
                state = SubagentState.FAILED
            summary = raw.get("summary") if isinstance(raw, dict) else None
            summary = str(summary)[:_MAX_RESULT_CHARS] if summary is not None else None
            error = raw.get("error") if isinstance(raw, dict) else None
            result = SubagentResult(
                record.handle,
                state,
                True,
                summary=summary,
                completed_at=time.time(),
                started_at=record.started_at,
                error_classification=None
                if state == SubagentState.SUCCEEDED
                else status.upper(),
                error_message=str(error)[:_MAX_RESULT_CHARS] if error else None,
                usage_metadata={"api_calls": raw.get("api_calls", 0)}
                if isinstance(raw, dict)
                else {},
                tool_execution_summary={
                    "duration_seconds": raw.get("duration_seconds", 0)
                }
                if isinstance(raw, dict)
                else {},
                launch_receipt=receipt_payload,
                execution_receipt=record.execution_receipt,
                execution_receipt_hash=(
                    record.execution_receipt.canonical_hash()
                    if record.execution_receipt is not None
                    else None
                ),
            )
        except Exception as exc:
            result = SubagentResult(
                record.handle,
                SubagentState.FAILED,
                True,
                started_at=record.started_at,
                completed_at=time.time(),
                error_classification=type(exc).__name__,
                error_message=str(exc)[:_MAX_RESULT_CHARS],
                launch_receipt=receipt_payload,
                execution_receipt=record.execution_receipt,
                execution_receipt_hash=(
                    record.execution_receipt.canonical_hash()
                    if record.execution_receipt is not None
                    else None
                ),
            )
        payload = dataclasses.asdict(result)
        payload.pop("result_hash", None)
        if payload.get("launch_receipt") is None:
            # Legacy launches keep their pre-profile hash input byte-for-byte;
            # profile launches fold the receipt into the result hash.
            payload.pop("launch_receipt", None)
        if payload.get("execution_receipt") is None:
            payload.pop("execution_receipt", None)
            payload.pop("execution_receipt_hash", None)
        result = dataclasses.replace(
            result,
            result_hash=hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )
        with _REGISTRY.lock:
            record.agent = None
            record.result = result
            record.state = result.terminal_state
            record.completed_at = result.completed_at
            record.updated_at = result.completed_at or time.time()

    @staticmethod
    def _run_process_child(
        record: _Record,
        goal: str,
        profile: ResolvedExecutionProfile,
    ) -> Mapping[str, Any]:
        from pathlib import Path

        from agent.subagent_broker_protocol import BrokerGrant, SubagentBroker
        from agent.subagent_process_integration import (
            ParentBrokerAdapter,
            launch_receipt_digest,
            strict_worker_runtime_mounts,
        )
        from agent.subagent_process_runner import (
            LinuxStrictConfig,
            ProcessRunSpec,
            run_owned_process,
        )
        from agent.subagent_tool_boundary import (
            EvoToolBoundaryError,
            classify_evo_tools,
        )

        if record.receipt is None or profile.workspace_root is None:
            raise SubagentLifecycleError(
                "process profile lacks pinned launch authority"
            )
        try:
            classify_evo_tools(record.agent._delegate_frozen_dispatch_entries)
        except EvoToolBoundaryError as exc:
            raise SubagentLifecycleError(str(exc)) from exc
        digest = launch_receipt_digest(record.receipt)
        broker = SubagentBroker(
            launch_receipt_digest=digest,
            grant=BrokerGrant(
                operations=frozenset({
                    "session.start",
                    "model.complete",
                    "tool.execute",
                }),
                workspace_root=profile.workspace_root,
            ),
        )
        recorder = SubagentExecutionRecorder(
            launch_receipt_digest=digest,
            backend=profile.execution_backend,
            containment_mode=(
                "portable-process-unconfined"
                if profile.execution_backend == "portable"
                else "linux-strict-bwrap-cgroup-v2"
            ),
            requested_cleanup=(
                ("process-group-reap", "cgroup-v2-kill-and-empty")
                if profile.execution_backend == "linux_strict"
                else ("process-group-reap",)
            ),
        )

        class ReceiptAdapter:
            def on_created(self, *, backend: str, confinement: str) -> None:
                del backend, confinement

            def on_started(self, *, root_pid: int) -> None:
                record.execution_receipt = recorder.mark_started(root_pid=root_pid)

            def on_terminal(self, *, state: str, result: Any) -> None:
                observed_cleanup = [
                    f"root_reaped={result.cleanup.root_reaped}",
                    f"process_group_empty={result.cleanup.process_group_empty}",
                ]
                if result.cleanup.cgroup_empty is not None:
                    observed_cleanup.append(
                        f"cgroup_empty={result.cleanup.cgroup_empty}"
                    )
                if result.cleanup.cgroup_removed is not None:
                    observed_cleanup.append(
                        f"cgroup_removed={result.cleanup.cgroup_removed}"
                    )
                diagnostics = [result.diagnostic] if result.diagnostic else []
                if result.stderr:
                    stderr_lines = [
                        line.strip()
                        for line in result.stderr.decode(
                            "utf-8", errors="replace"
                        ).splitlines()
                        if line.strip()
                    ]
                    if stderr_lines:
                        diagnostic = "".join(
                            character if character.isprintable() else " "
                            for character in stderr_lines[-1]
                        )
                        diagnostics.append(
                            f"worker stderr: {' '.join(diagnostic.split())[:1000]}"
                        )
                evidence = {
                    "exit_code": result.exit_code,
                    "term_signal": result.signal,
                    "broker_transcript_digest": broker.transcript_digest(),
                    "observed_cleanup": tuple(observed_cleanup),
                    "containment_evidence": (result.confinement,),
                    "diagnostics": tuple(diagnostics),
                }
                marker = getattr(recorder, f"mark_{state.lower()}")
                record.execution_receipt = marker(**evidence)

        adapter = ParentBrokerAdapter(
            broker=broker,
            child=record.agent,
            profile=profile,
            task=goal,
        )
        workspace = Path(profile.workspace_root)
        backend = (
            "portable" if profile.execution_backend == "portable" else "linux-strict"
        )
        worker_path = (
            Path(__file__).with_name("subagent_worker_main.py").resolve(strict=True)
        )
        runtime_mounts = ()
        if backend == "linux-strict":
            runtime_mounts = strict_worker_runtime_mounts()
        spec = ProcessRunSpec(
            executable=sys.executable,
            argv=(str(worker_path),),
            cwd=workspace,
            workspace=workspace,
            capability_secret=broker.reveal_secret_for_transport(),
            capability_id=broker.capability_id,
            launch_receipt_digest=digest,
            backend=backend,
            timeout_seconds=profile.timeout_seconds or 300.0,
            broker=adapter,
            receipt=ReceiptAdapter(),
            runtime_mounts=runtime_mounts,
            strict=(
                LinuxStrictConfig(
                    cgroup_parent=Path(profile.cgroup_parent or ""),
                    sandbox_workspace=workspace,
                )
                if backend == "linux-strict"
                else None
            ),
        )
        result = run_owned_process(spec)
        broker.close("owned process completed")
        if result.state != "SUCCEEDED":
            worker_errors = [result.diagnostic] if result.diagnostic else []
            if result.stderr:
                stderr_lines = [
                    line.strip()
                    for line in result.stderr.decode(
                        "utf-8", errors="replace"
                    ).splitlines()
                    if line.strip()
                ]
                if stderr_lines:
                    diagnostic = "".join(
                        character if character.isprintable() else " "
                        for character in stderr_lines[-1]
                    )
                    worker_errors.append(
                        f"worker stderr: {' '.join(diagnostic.split())[:1000]}"
                    )
            return {
                "status": "error",
                "error": "; ".join(worker_errors) or result.state,
                "api_calls": 0,
                "duration_seconds": 0,
            }
        try:
            payload = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SubagentLifecycleError(
                "worker returned malformed bounded JSON"
            ) from exc
        if not isinstance(payload, Mapping) or set(payload) != {
            "summary",
            "iterations",
        }:
            raise SubagentLifecycleError("worker result has an invalid shape")
        return {
            "status": "completed",
            "summary": payload["summary"],
            "api_calls": payload["iterations"],
            "duration_seconds": 0,
        }

    @staticmethod
    def _capability(
        subagent_id: str, parent_session_id: Optional[str], created_at: float
    ) -> str:
        value = f"{subagent_id}|{parent_session_id or ''}|{created_at:.6f}".encode()
        return hmac.new(_SECRET, value, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_request(request: SubagentLaunchRequest, parent: Any) -> None:
        if (
            not isinstance(request, SubagentLaunchRequest)
            or not isinstance(request.goal, str)
            or not request.goal.strip()
            or len(request.goal) > _MAX_GOAL_CHARS
        ):
            raise SubagentLifecycleError(
                "goal must be a non-empty string of at most 16000 characters."
            )
        if request.context is not None and (
            not isinstance(request.context, str)
            or len(request.context) > _MAX_CONTEXT_CHARS
        ):
            raise SubagentLifecycleError(
                "context must be a string of at most 32000 characters."
            )
        if request.role not in {"leaf", "orchestrator"}:
            raise SubagentLifecycleError("role must be 'leaf' or 'orchestrator'.")
        if request.timeout_seconds is not None:
            raise SubagentLifecycleError(
                "Per-launch timeout is not supported; configure delegation timeout explicitly."
            )
        if request.working_directory is not None:
            raise SubagentLifecycleError(
                "working_directory is not supported because Hermes delegates use isolated task environments."
            )
        if request.blocked_tools:
            raise SubagentLifecycleError(
                "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."
            )
        try:
            metadata_bytes = len(
                json.dumps(dict(request.metadata), sort_keys=True).encode()
            )
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("metadata must be JSON-serializable.") from exc
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise SubagentLifecycleError("metadata exceeds 8192 bytes.")
        if request.allowed_toolsets:
            from toolsets import TOOLSETS

            unknown = set(request.allowed_toolsets) - set(TOOLSETS)
            if unknown:
                raise SubagentLifecycleError(
                    f"Unknown toolsets: {', '.join(sorted(unknown))}."
                )
            enabled = getattr(parent, "enabled_toolsets", None)
            if enabled is not None and not set(request.allowed_toolsets).issubset(
                set(enabled)
            ):
                raise SubagentLifecycleError(
                    "Requested toolsets would broaden parent permissions."
                )
