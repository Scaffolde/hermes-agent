"""Contract tests for the public plugin subagent lifecycle API."""

import dataclasses
import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import tools.delegate_tool  # noqa: F401 - establish registry before fake snapshots

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    bind_subagent_parent,
    get_active_subagent_parent,
)
from agent.subagent_execution_profiles import ResolvedExecutionProfile


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.api_mode = "chat_completions"
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None
        self.session_id = f"session-{ident}"
        self.valid_tool_names = {"read_file", "write_file"}
        self.tools = [
            {"type": "function", "function": {"name": name}}
            for name in sorted(self.valid_tool_names)
        ]
        from tools.registry import registry

        self._delegate_tool_registry_generation = registry.generation()
        self.ephemeral_system_prompt = "host-built system prompt"
        self._publish_deferred_launch: Any = None
        self._publish_deferred_context_session_start: Any = None
        self.closed = False

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason

    def close(self):
        self.closed = True


def _profile(**overrides):
    profile = ResolvedExecutionProfile(
        profile_id="reviewer",
        role="leaf",
        allowed_toolsets=("file",),
        expected_tool_names=frozenset({"read_file", "write_file"}),
        protocol_file="protocols/reviewer.md",
        protocol_text="# Reviewer protocol\n",
        protocol_sha256=hashlib.sha256(b"# Reviewer protocol\n").hexdigest(),
        allow_root=True,
        allowed_child_profiles=(),
        timeout_seconds=None,
    )
    return dataclasses.replace(profile, **overrides)


def test_process_profile_refuses_unsupported_api_mode_before_spawn(
    monkeypatch, tmp_path
):
    parent = SimpleNamespace(session_id="process-refusal", enabled_toolsets=["file"])
    child = FakeChild("unsupported-mode")
    child.api_mode = "codex_responses"
    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda _id: _profile(
            execution_backend="portable", workspace_root=str(tmp_path)
        ),
    )

    def build(**_kwargs):
        from tools.registry import registry

        child._delegate_tool_registry_generation = registry.generation()
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    spawned = Mock()
    monkeypatch.setattr("agent.subagent_process_runner.run_owned_process", spawned)

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(
        SubagentLifecycleError, match="only api_mode 'chat_completions'"
    ):
        service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))
    assert child.closed is True
    spawned.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "hostname"),
    [
        ("xai", None),
        ("xai-oauth", None),
        ("custom", "api.x.ai"),
    ],
)
def test_exact_profile_refuses_xai_responses_schema_rewrite(
    monkeypatch, provider, hostname
):
    parent = SimpleNamespace(session_id="xai-refusal", enabled_toolsets=["file"])
    child = FakeChild("xai-responses")
    child.api_mode = "codex_responses"
    child.provider = provider
    setattr(child, "_base_url_hostname", hostname)
    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile", lambda _id: _profile()
    )

    def build(**_kwargs):
        from tools.registry import registry

        child._delegate_tool_registry_generation = registry.generation()
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(
        SubagentLifecycleError,
        match="xAI Responses because its provider wire path rewrites tool schemas",
    ):
        service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))
    assert child.closed is True


def test_profile_workspace_binding_rejects_caller_selected_root_before_spawn(
    monkeypatch, tmp_path
):
    parent = SimpleNamespace(session_id="workspace-refusal", enabled_toolsets=["file"])
    host_workspace = tmp_path / "host-owned"
    caller_workspace = tmp_path / "caller-selected"
    host_workspace.mkdir()
    caller_workspace.mkdir()
    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda _id: _profile(workspace_root=str(host_workspace)),
    )
    build = Mock()
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(
        SubagentLifecycleError,
        match="does not match the host-owned execution profile workspace",
    ):
        service.launch(
            SubagentLaunchRequest(
                goal="review",
                profile_id="reviewer",
                required_workspace_root=str(caller_workspace),
            )
        )
    build.assert_not_called()


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)


def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)


def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", lambda **_kwargs: child
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"


def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None


def test_profile_launch_is_host_resolved_and_receipted(monkeypatch):
    parent = SimpleNamespace(session_id="profile-parent", enabled_toolsets=["file"])
    child = FakeChild("profile")
    child.tools[0]["function"]["x-launch-marker"] = "pinned"
    publish = Mock()
    child._publish_deferred_launch = publish
    publish_context_start = Mock()
    child._publish_deferred_context_session_start = publish_context_start
    observed = {}

    def build(**kwargs):
        from tools.registry import registry

        observed.update(kwargs)
        child._delegate_tool_registry_generation = registry.generation()
        return child

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda profile_id: _profile(profile_id=profile_id),
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)

    def run_single(_task_index, goal, *_args, **_kwargs):
        observed["user_task"] = goal
        return {"status": "completed", "summary": "ok"}

    monkeypatch.setattr("tools.delegate_tool._run_single_child", run_single)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(
        SubagentLaunchRequest(
            goal="review", context="experiment 7", profile_id="reviewer"
        )
    )
    receipt = service.describe(handle)
    assert receipt is not None
    assert receipt.profile_id == "reviewer"
    assert receipt.child_session_id == "session-profile"
    assert receipt.role == "leaf"
    assert receipt.resolved_tool_names == ("read_file", "write_file")
    assert receipt.protocol_sha256 == _profile().protocol_sha256
    assert receipt.goal_sha256 == hashlib.sha256(b"review").hexdigest()
    assert not hasattr(receipt, "system_prompt_sha256")
    assert set(child._delegate_frozen_dispatch_entries) == {
        "read_file",
        "write_file",
    }
    from tools.registry import registry

    frozen_dispatch = getattr(child, "_delegate_frozen_dispatch_entries")
    frozen_read_schema = registry.get_snapshot_schema(frozen_dispatch, "read_file")
    assert frozen_read_schema is not None
    assert frozen_read_schema["x-launch-marker"] == "pinned"
    assert observed["toolsets"] == ["file"]
    assert observed["role"] == "leaf"
    assert observed["inherit_mcp"] is False
    assert observed["context"] == "experiment 7"
    assert observed["system_prompt_override"] == "# Reviewer protocol\n"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(receipt, "role", "orchestrator")

    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED
    assert observed["user_task"] == "review\n\nContext:\nexperiment 7"
    publish.assert_called_once_with()
    publish_context_start.assert_called_once_with()
    result = service.result(handle)
    assert result.launch_receipt == receipt
    assert result.result_hash


def test_profile_blocked_tools_are_removed_before_exact_freeze(monkeypatch):
    parent = SimpleNamespace(session_id="profile-parent", enabled_toolsets=["file"])
    child = FakeChild("sa-blocked")
    child.valid_tool_names.add("patch")
    child.tools.append({"type": "function", "function": {"name": "patch"}})
    from tools.registry import registry

    def build(**_kwargs):
        child._delegate_tool_registry_generation = registry.generation()
        return child

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda _profile_id: _profile(blocked_tools=frozenset({"patch"})),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent",
        build,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        },
    )
    service = SubagentLifecycleService(lambda: parent)

    handle = service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))
    receipt = service.describe(handle)

    assert receipt is not None
    assert receipt.resolved_tool_names == ("read_file", "write_file")
    assert child.valid_tool_names == {"read_file", "write_file"}
    assert [tool["function"]["name"] for tool in child.tools] == [
        "read_file",
        "write_file",
    ]
    assert set(child._delegate_frozen_tool_names) == {"read_file", "write_file"}


@pytest.mark.parametrize("failure", ["tools", "role"])
def test_profile_launch_rejects_effective_contract_drift(monkeypatch, failure):
    parent = SimpleNamespace(session_id=f"profile-{failure}", enabled_toolsets=["file"])
    child = FakeChild(f"bad-{failure}")
    publish = Mock()
    child._publish_deferred_launch = publish
    publish_context_start = Mock()
    child._publish_deferred_context_session_start = publish_context_start
    if failure == "tools":
        child.valid_tool_names.add("terminal")
    else:
        child._delegate_role = "orchestrator"

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile", lambda _id: _profile()
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", lambda **_kwargs: child
    )

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(
        SubagentLifecycleError, match="exact tool|never silently degrade"
    ):
        service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))
    assert child.closed is True
    publish.assert_not_called()
    publish_context_start.assert_not_called()


def test_profile_rejects_dynamic_context_engine_dispatch():
    child = FakeChild("dynamic-route")
    child.valid_tool_names = {"lcm_grep"}
    child.tools = [{"type": "function", "function": {"name": "lcm_grep"}}]
    setattr(child, "_context_engine_tool_names", {"lcm_grep"})
    profile = dataclasses.replace(
        _profile(), expected_tool_names=frozenset({"lcm_grep"})
    )

    with pytest.raises(
        SubagentLifecycleError, match="without frozen dispatch adapters"
    ):
        SubagentLifecycleService._enforce_profile_contract(
            child,
            profile,
            SubagentLaunchRequest(goal="review", profile_id="reviewer"),
            None,
            time.time(),
        )


def test_profile_freeze_serializes_with_inflight_mcp_publish(monkeypatch):
    import model_tools
    from tools import mcp_tool

    child = FakeChild("mcp-race")
    setattr(child, "enabled_toolsets", ["file"])
    setattr(child, "disabled_toolsets", [])
    setattr(child, "_memory_manager", None)
    setattr(child, "_context_engine_tool_names", set())
    setattr(child, "context_compressor", SimpleNamespace(get_tool_schemas=lambda: []))
    setattr(child, "_tool_snapshot_generation", -1)
    entered = threading.Event()
    release = threading.Event()
    enforced = threading.Event()
    observed = {}

    class GateLock:
        def __init__(self):
            self._lock = threading.Lock()
            self._refresh_entries = 0

        def __enter__(self):
            self._lock.acquire()
            if threading.current_thread().name == "mcp-refresh":
                self._refresh_entries += 1
                if self._refresh_entries == 2:
                    entered.set()
                    assert release.wait(timeout=2)
            return self

        def __exit__(self, *_args):
            self._lock.release()

    gate_lock = GateLock()
    monkeypatch.setattr(mcp_tool, "_agent_tools_lock", gate_lock)
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kwargs: [
            {"type": "function", "function": {"name": name}}
            for name in ("read_file", "write_file", "terminal")
        ],
    )

    refresh = threading.Thread(
        name="mcp-refresh", target=lambda: mcp_tool.refresh_agent_mcp_tools(child)
    )
    refresh.start()
    assert entered.wait(timeout=1)

    def enforce():
        try:
            SubagentLifecycleService._enforce_profile_contract(
                child,
                _profile(),
                SubagentLaunchRequest(goal="review", profile_id="reviewer"),
                None,
                time.time(),
            )
        except Exception as exc:
            observed["error"] = exc
        finally:
            enforced.set()

    verifier = threading.Thread(name="profile-freeze", target=enforce)
    verifier.start()
    time.sleep(0.02)
    assert not enforced.is_set()

    release.set()
    refresh.join(timeout=2)
    verifier.join(timeout=2)

    assert not refresh.is_alive()
    assert not verifier.is_alive()
    assert isinstance(observed.get("error"), SubagentLifecycleError)
    assert "exact tool contract" in str(observed["error"])
    assert not hasattr(child, "_delegate_frozen_tool_names")


def test_profile_submission_failure_emits_no_start_observers(monkeypatch):
    parent = SimpleNamespace(session_id="profile-submit", enabled_toolsets=["file"])
    child = FakeChild("submit-failure")
    publish = Mock()
    publish_context_start = Mock()
    child._publish_deferred_launch = publish
    child._publish_deferred_context_session_start = publish_context_start

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile", lambda _id: _profile()
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", lambda **_kwargs: child
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle._EXECUTOR.submit",
        Mock(side_effect=RuntimeError("executor unavailable")),
    )

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(RuntimeError, match="executor unavailable"):
        service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))

    publish.assert_not_called()
    publish_context_start.assert_not_called()
    assert child.closed is True


def test_profile_worker_waits_for_deferred_observers(monkeypatch):
    parent = SimpleNamespace(session_id="profile-gate", enabled_toolsets=["file"])
    child = FakeChild("start-gate")
    order = []
    child._publish_deferred_context_session_start = lambda: order.append("context")
    child._publish_deferred_launch = lambda: order.append("launch")

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile", lambda _id: _profile()
    )

    def build(**_kwargs):
        from tools.registry import registry

        child._delegate_tool_registry_generation = registry.generation()
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: (
            order.append("run") or {"status": "completed", "summary": "ok"}
        ),
    )

    def submit(fn, *args):
        future = Future()

        def target():
            try:
                fn(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        thread = threading.Thread(target=target)
        thread.start()
        time.sleep(0.02)
        assert "run" not in order
        return future

    monkeypatch.setattr("agent.subagent_lifecycle._EXECUTOR.submit", submit)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="review", profile_id="reviewer"))

    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED
    assert order == ["context", "launch", "run"]


def test_profile_launch_rejects_caller_policy_and_combined_context(monkeypatch):
    parent = SimpleNamespace(session_id="profile-inputs", enabled_toolsets=["file"])
    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda _id: _profile(protocol_text="p" * 31_999),
    )
    service = SubagentLifecycleService(lambda: parent)

    with pytest.raises(SubagentLifecycleError, match="profile-owned"):
        service.launch(
            SubagentLaunchRequest(
                goal="review", profile_id="reviewer", allowed_toolsets=("file",)
            )
        )
    with pytest.raises(SubagentLifecycleError, match="exceeds 32000"):
        service.launch(
            SubagentLaunchRequest(goal="review", context="extra", profile_id="reviewer")
        )


def test_profile_launch_reserves_executor_capacity_during_child_build(monkeypatch):
    """Concurrent profile construction must count against the fail-fast gate."""
    import agent.subagent_lifecycle as lifecycle_module

    parent = SimpleNamespace(session_id="profile-capacity", enabled_toolsets=["file"])
    entered = threading.Event()
    release = threading.Event()
    build_calls = 0

    def build(**_kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return FakeChild(f"capacity-{build_calls}")

    monkeypatch.setattr(lifecycle_module, "_EXECUTOR_MAX_WORKERS", 1)
    monkeypatch.setattr(
        lifecycle_module, "resolve_execution_profile", lambda _id: _profile()
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {"status": "completed", "summary": "ok"},
    )

    service = SubagentLifecycleService(lambda: parent)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            service.launch,
            SubagentLaunchRequest(goal="first", profile_id="reviewer"),
        )
        assert entered.wait(timeout=1)
        try:
            with pytest.raises(SubagentLifecycleError, match="saturated"):
                service.launch(
                    SubagentLaunchRequest(goal="second", profile_id="reviewer")
                )
            assert build_calls == 1
        finally:
            release.set()
        handle = first.result(timeout=2)

    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED


def test_profile_launch_reserves_correlation_during_child_build(monkeypatch):
    """A concurrent duplicate correlation must fail before child construction."""
    import agent.subagent_lifecycle as lifecycle_module

    parent = SimpleNamespace(
        session_id="profile-correlation", enabled_toolsets=["file"]
    )
    entered = threading.Event()
    release = threading.Event()
    build_calls = 0

    def build(**_kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return FakeChild(f"correlation-{build_calls}")

    monkeypatch.setattr(
        lifecycle_module, "resolve_execution_profile", lambda _id: _profile()
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {"status": "completed", "summary": "ok"},
    )

    request = SubagentLaunchRequest(
        goal="review", profile_id="reviewer", correlation_id="same-work"
    )
    service = SubagentLifecycleService(lambda: parent)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service.launch, request)
        assert entered.wait(timeout=1)
        try:
            with pytest.raises(
                SubagentLifecycleError, match="Duplicate correlation_id"
            ):
                service.launch(request)
            assert build_calls == 1
        finally:
            release.set()
        handle = first.result(timeout=2)

    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED


def test_profile_parent_cannot_launch_unprofiled_child(monkeypatch):
    parent = SimpleNamespace(
        session_id="profile-parent",
        enabled_toolsets=["file"],
        _execution_profile_id="candidate",
    )
    build = Mock()
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)

    service = SubagentLifecycleService(lambda: parent)
    with pytest.raises(SubagentLifecycleError, match="named execution profile"):
        service.launch(SubagentLaunchRequest(goal="bypass the profile graph"))
    build.assert_not_called()


def test_profile_deadline_is_forwarded_to_child_lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="profile-deadline", enabled_toolsets=["file"])
    observed = {}

    monkeypatch.setattr(
        "agent.subagent_lifecycle.resolve_execution_profile",
        lambda _id: _profile(timeout_seconds=0.25),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent",
        lambda **_kwargs: FakeChild("deadline"),
    )

    def run(_index, _goal, _child, _parent, *, timeout_override=None):
        observed["timeout_override"] = timeout_override
        return {
            "status": "timeout",
            "summary": None,
            "error_classification": "TIMEOUT",
            "error_message": "deadline exceeded",
        }

    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(
        SubagentLaunchRequest(goal="time bounded", profile_id="reviewer")
    )
    status = service.wait(handle, timeout_seconds=1)
    result = service.result(handle)

    assert observed == {"timeout_override": 0.25}
    assert status.state is SubagentState.FAILED
    assert result.error_classification == "TIMEOUT"
