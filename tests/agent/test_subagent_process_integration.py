import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent.subagent_process_integration as process_integration
from agent.subagent_broker_protocol import BrokerGrant, SubagentBroker, canonical_json
from agent.subagent_process_integration import (
    ParentBrokerAdapter,
    ProcessIntegrationError,
    strict_worker_runtime_mounts,
    strict_worker_runtime_path,
)
from agent.subagent_worker_main import (
    BrokerFrameError,
    _dispatch_local,
    run_worker_loop,
)
from agent.subagent_process_runner import ProcessRunSpec, run_owned_process


@pytest.fixture(autouse=True)
def _resolved_scaffolde_profiles(monkeypatch):
    monkeypatch.setattr(
        process_integration,
        "resolve_execution_profile",
        lambda profile_id: SimpleNamespace(
            profile_id=profile_id,
            protocol_sha256="a" * 64,
        ),
    )


def _valid_broker_arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "workspace": str(tmp_path),
        "role": "verifier",
        "experiment_id": "exp_1001",
        "phase": "post",
        "attempt_n": 1,
    }


def _fixture(tmp_path: Path, responses):
    launch_digest = hashlib.sha256(b"launch").hexdigest()
    broker = SubagentBroker(
        launch_receipt_digest=launch_digest,
        grant=BrokerGrant(
            operations=frozenset({"session.start", "model.complete", "tool.execute"}),
            workspace_root=str(tmp_path),
        ),
    )

    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(usage={"prompt_tokens": 1})

    completions = Completions()
    child = SimpleNamespace(
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        _delegate_frozen_dispatch_entries={"read_file": object()},
        _subagent_id="child-1",
        session_id="session-1",
    )
    child._build_api_kwargs = lambda messages, tools_for_api: {
        "messages": messages,
        "tools": tools_for_api,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    child._get_transport = lambda: SimpleNamespace(
        normalize_response=lambda _response: responses.pop(0)
    )
    profile = SimpleNamespace(
        profile_id="scaffolde-evo-candidate-v1",
        protocol_sha256="a" * 64,
        protocol_text="strict protocol",
        max_process_iterations=4,
        workspace_root=str(tmp_path),
    )
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=profile, task="inspect task"
    )
    return broker, adapter, child, completions, launch_digest


def _brokered_fixture(tmp_path: Path):
    broker, old_adapter, child, completions, digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_agent_dispatch"
    child.tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
    ]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="nested task"
    )
    return broker, adapter, child, completions, digest


def _valid_broker_attestation() -> dict[str, object]:
    return {
        "version": 1,
        "role": "verifier",
        "profile_id": "scaffolde-evo-verifier-v1",
        "protocol_sha256": "a" * 64,
        "ok": True,
        "request": {
            "experiment_id": "exp_1001",
            "phase": "post",
            "attempt_n": 1,
        },
        "completion": {
            "result_hash": "b" * 64,
            "execution_receipt_hash": "c" * 64,
        },
        "outcome": {"passed": True, "verdict": "pass"},
        "evidence_sha256": hashlib.sha256(
            canonical_json({"verified": True}).encode()
        ).hexdigest(),
    }


def _valid_broker_result(
    *, attestation: dict[str, object] | None = None, **extra: object
) -> dict[str, object]:
    return {
        **extra,
        "broker_attestation": attestation or _valid_broker_attestation(),
        "completion": {"result_hash": "b" * 64},
        "execution_receipt_hash": "c" * 64,
        "evidence": {"verified": True},
    }


def _valid_run_attestation() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "evo-run",
        "role": "candidate-worker",
        "profile_id": "scaffolde-evo-candidate-v1",
        "protocol_sha256": "a" * 64,
        "ok": True,
        "request": {"experiment_id": "exp_1001", "attempt_n": 1},
        "completion": {
            "result_hash": "b" * 64,
            "execution_receipt_hash": "c" * 64,
        },
        "outcome": {"status": "committed", "exit_code": 0},
        "attempt_execution": {
            "backend": "linux-strict",
            "containment_mode": "linux-strict-bwrap-cgroup-v2",
            "authority_sha256": "d" * 64,
            "executable_sha256": "e" * 64,
            "cleanup_sha256": "f" * 64,
        },
        "evidence_sha256": hashlib.sha256(
            canonical_json({"verified": True}).encode()
        ).hexdigest(),
    }


def _valid_run_result_payload() -> dict[str, Any]:
    result = {
        "experiment_id": "exp_1001",
        "attempt_n": 1,
        "status": "committed",
        "exit_code": 0,
    }
    cleanup = {
        "root_reaped": True,
        "process_group_empty": True,
        "cgroup_kill_sent": False,
        "cgroup_empty": True,
        "cgroup_removed": True,
        "broker_quiesced": None,
    }
    execution_receipt = {
        "version": 2,
        "runner": "scaffolde-evo-run",
        "backend": "linux-strict",
        "containment_mode": "linux-strict-bwrap-cgroup-v2",
        "state": "SUCCEEDED",
        "authority_sha256": "d" * 64,
        "executable_sha256": "e" * 64,
        "argv_sha256": "1" * 64,
        "root_pid": 4242,
        "exit_code": 0,
        "timed_out": False,
        "cancelled": False,
        "cleanup": cleanup,
    }
    evidence = {"verified": True}
    attestation = _valid_run_attestation()
    result_hash = hashlib.sha256(canonical_json(result).encode()).hexdigest()
    receipt_hash = hashlib.sha256(
        canonical_json(execution_receipt).encode()
    ).hexdigest()
    attestation["completion"] = {
        "result_hash": result_hash,
        "execution_receipt_hash": receipt_hash,
    }
    attestation["attempt_execution"] = {
        "backend": "linux-strict",
        "containment_mode": "linux-strict-bwrap-cgroup-v2",
        "authority_sha256": "d" * 64,
        "executable_sha256": "e" * 64,
        "cleanup_sha256": hashlib.sha256(canonical_json(cleanup).encode()).hexdigest(),
    }
    return {
        "ok": True,
        "broker_attestation": attestation,
        "completion": {"result_hash": result_hash},
        "result": result,
        "execution_receipt": execution_receipt,
        "execution_receipt_hash": receipt_hash,
        "evidence": evidence,
    }


def test_parent_cancel_has_a_real_deadline_for_an_admitted_operation(tmp_path):
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [])
    assert adapter._operation_lock.acquire(timeout=0.1)
    started = time.monotonic()
    try:
        assert adapter.cancel(timeout_seconds=0.01) is False
    finally:
        adapter._operation_lock.release()
    assert time.monotonic() - started < 0.2


def test_parent_cancel_defers_claim_freeze_until_admitted_operation_quiesces(
    tmp_path,
):
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    adapter._reserve_claim_capacity()
    assert adapter._operation_lock.acquire(timeout=0.1)
    try:
        assert adapter.cancel(timeout_seconds=0.01) is False
    finally:
        adapter._operation_lock.release()

    assert adapter.claims_frozen is False
    assert adapter.claim_cleanup_failure is None
    adapter.finalize()
    assert adapter.claims_frozen is True
    with pytest.raises(ProcessIntegrationError, match="claim accounting is frozen"):
        adapter._record_tool_execution_claim(
            name="scaffolde_evo_agent_dispatch",
            arguments_sha256="a" * 64,
            result_sha256="b" * 64,
            public_attestation_json="{}",
        )
    assert adapter.tool_execution_claims == ()


def test_parent_cancel_reports_quiesced_when_no_operation_is_admitted(tmp_path):
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [])
    assert adapter.cancel(timeout_seconds=0.01) is True
    assert adapter._cancellation_event.is_set()


def test_parent_rejects_provider_schema_not_bound_to_launch_receipt(tmp_path):
    broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])

    with pytest.raises(ProcessIntegrationError, match="launch receipt"):
        ParentBrokerAdapter(
            broker=broker,
            child=child,
            profile=old_adapter.profile,
            task="schema mismatch",
            expected_tool_schema_sha256="f" * 64,
        )


def test_parent_binds_session_and_completion_to_provider_effective_tool_schema(
    tmp_path,
):
    normalized = SimpleNamespace(
        content="finished", finish_reason="stop", reasoning=None, tool_calls=[]
    )
    broker, old_adapter, child, completions, _digest = _fixture(tmp_path, [normalized])
    effective_tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        }
    ]
    child._build_api_kwargs = lambda messages, tools_for_api: {
        "messages": messages,
        "tools": effective_tools,
    }
    adapter = ParentBrokerAdapter(
        broker=broker,
        child=child,
        profile=old_adapter.profile,
        task="effective schema task",
    )

    session = adapter._dispatch("session.start", {})
    adapter._dispatch(
        "model.complete", {"messages": [{"role": "user", "content": "go"}]}
    )

    assert session["tools"] == effective_tools
    assert session[
        "tool_schema_digest"
    ] == process_integration.exact_tool_schema_digest(effective_tools)
    assert completions.kwargs["tools"] == effective_tools


def test_session_start_rejects_provider_schema_drift_after_adapter_construction(
    tmp_path,
):
    _broker, adapter, child, _completions, _digest = _fixture(tmp_path, [])
    child.tools[0]["function"]["parameters"]["additionalProperties"] = False

    with pytest.raises(ProcessIntegrationError, match="schema changed after launch"):
        adapter._dispatch("session.start", {})


def test_model_completion_rejects_provider_wrapper_drift(tmp_path):
    normalized = SimpleNamespace(
        content="finished", finish_reason="stop", reasoning=None, tool_calls=[]
    )
    _broker, adapter, child, completions, _digest = _fixture(tmp_path, [normalized])
    original_builder = child._build_api_kwargs

    def drifted_builder(messages, *, tools_for_api):
        kwargs = original_builder(messages, tools_for_api=tools_for_api)
        kwargs["tools"][0]["strict"] = False
        return kwargs

    child._build_api_kwargs = drifted_builder

    with pytest.raises(ProcessIntegrationError, match="schema changed after launch"):
        adapter._dispatch("model.complete", {"messages": []})
    assert not hasattr(completions, "kwargs")


def test_worker_loop_success_uses_authenticated_broker_and_non_streaming(tmp_path):
    normalized = SimpleNamespace(
        content="finished", finish_reason="stop", reasoning=None, tool_calls=[]
    )
    broker, adapter, _child, completions, digest = _fixture(tmp_path, [normalized])
    host, worker = socket.socketpair()
    stop = threading.Event()
    errors = []

    def serve():
        try:
            adapter.serve(host, root_pid=123, stop_requested=stop)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    result = run_worker_loop(
        worker,
        broker.reveal_secret_for_transport(),
        capability_id=broker.capability_id,
        launch_receipt_digest=digest,
    )
    assert result == {"summary": "finished", "iterations": 1}
    assert completions.kwargs["stream"] is False
    assert "stream_options" not in completions.kwargs
    stop.set()
    worker.close()
    host.close()


def test_parent_never_dispatches_worker_local_tool(tmp_path, monkeypatch):
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [])
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: pytest.fail("parent must not execute local tools"),
    )
    with pytest.raises(ProcessIntegrationError, match="not host-brokered"):
        adapter._dispatch(
            "tool.execute",
            {"id": "1", "name": "terminal", "arguments": {"command": "pwd"}},
        )


def test_worker_loop_executes_exact_tool_call_then_finishes(tmp_path, monkeypatch):
    target = tmp_path / "notes" / "a.md"
    target.parent.mkdir()
    target.write_text("file contents")
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"notes/a.md"}'),
    )
    responses = [
        SimpleNamespace(
            content=None, finish_reason="tool_calls", reasoning=None, tool_calls=[call]
        ),
        SimpleNamespace(
            content="reviewed", finish_reason="stop", reasoning=None, tool_calls=[]
        ),
    ]
    broker, adapter, _child, _completions, digest = _fixture(tmp_path, responses)
    monkeypatch.setattr(
        "agent.subagent_worker_main._dispatch_local",
        lambda *_args, **_kwargs: pytest.fail(
            "brokered reads must not execute locally"
        ),
    )

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "secure brokered reads must not reopen through the registry"
        ),
    )
    host, worker = socket.socketpair()
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: adapter.serve(host, root_pid=123, stop_requested=stop),
        daemon=True,
    )
    thread.start()
    result = run_worker_loop(
        worker,
        broker.reveal_secret_for_transport(),
        capability_id=broker.capability_id,
        launch_receipt_digest=digest,
    )
    assert result == {"summary": "reviewed", "iterations": 2}
    assert "1|file contents" in str(_completions.kwargs["messages"][-1]["content"])
    assert adapter.tool_execution_claims == ()
    stop.set()
    worker.shutdown(socket.SHUT_RDWR)
    thread.join(timeout=1)
    worker.close()
    host.close()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", {"path": "../outside.txt"}),
        ("read_file", {"path": "missing.txt"}),
    ],
)
def test_brokered_reads_reject_escape_and_unavailable_paths(
    tmp_path, monkeypatch, name, arguments
):
    _broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    child.tools = [{"type": "function", "function": {"name": name, "parameters": {}}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=old_adapter.broker,
        child=child,
        profile=old_adapter.profile,
        task=old_adapter.task,
    )
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "rejected paths must not reach the registry"
        ),
    )
    assert adapter._dispatch(
        "tool.execute", {"id": "read-1", "name": name, "arguments": arguments}
    ) == {
        "result": {
            "ok": False,
            "error_code": "workspace_path_invalid",
            "error": "read_file path must be available under the profile workspace",
        }
    }


def test_brokered_read_rejects_symlink_escape(tmp_path, monkeypatch):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [])
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "symlink escapes must not reach the registry"
        ),
    )
    assert adapter._dispatch(
        "tool.execute",
        {
            "id": "read-1",
            "name": "read_file",
            "arguments": {"path": "escape/secret.txt"},
        },
    ) == {
        "result": {
            "ok": False,
            "error_code": "workspace_path_invalid",
            "error": "read_file path must be available under the profile workspace",
        }
    }


def test_unknown_frozen_tool_never_reaches_live_registry(tmp_path, monkeypatch):
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [])
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch",
        lambda *a, **k: pytest.fail("live registry must not be used"),
    )
    with pytest.raises(ProcessIntegrationError, match="not host-brokered"):
        adapter._dispatch(
            "tool.execute",
            {"id": "1", "name": "terminal", "arguments": {"command": "pwd"}},
        )


def test_real_separate_process_worker_returns_one_bounded_json_payload(tmp_path):
    normalized = SimpleNamespace(
        content="process complete", finish_reason="stop", reasoning=None, tool_calls=[]
    )
    broker, adapter, _child, _completions, digest = _fixture(tmp_path, [normalized])
    worker_path = Path(__file__).parents[2] / "agent" / "subagent_worker_main.py"
    result = run_owned_process(
        ProcessRunSpec(
            executable=sys.executable,
            argv=(str(worker_path),),
            cwd=tmp_path,
            workspace=tmp_path,
            capability_secret=broker.reveal_secret_for_transport(),
            capability_id=broker.capability_id,
            launch_receipt_digest=digest,
            capability_fd_arg="--capability-fd",
            broker=adapter,
            timeout_seconds=5,
        )
    )
    assert result.state == "SUCCEEDED", result
    assert result.stderr == b""
    assert result.stdout == b'{"iterations":1,"summary":"process complete"}'


def test_unknown_tool_classification_refuses_before_runner_spawn(tmp_path):
    broker, _adapter, child, _completions, _digest = _fixture(tmp_path, [])
    child._delegate_frozen_dispatch_entries = {"browser_navigate": object()}
    child.tools = [
        {"type": "function", "function": {"name": "browser_navigate", "parameters": {}}}
    ]
    with pytest.raises(ProcessIntegrationError, match="classification is undefined"):
        ParentBrokerAdapter(
            broker=broker,
            child=child,
            profile=SimpleNamespace(
                protocol_text="strict",
                max_process_iterations=2,
                workspace_root=str(tmp_path),
            ),
            task="task",
        )


def test_normalized_usage_wins_over_raw_provider_usage(tmp_path):
    normalized = SimpleNamespace(
        content="ok",
        finish_reason="stop",
        reasoning=None,
        tool_calls=[],
        usage={"total": 7},
    )
    _broker, adapter, _child, _completions, _digest = _fixture(tmp_path, [normalized])
    body = adapter._dispatch("model.complete", {"messages": []})
    assert body["usage"] == {"total": 7}


def test_real_worker_terminal_executes_below_worker_not_parent_broker(
    tmp_path, monkeypatch
):
    call = SimpleNamespace(
        id="call-terminal",
        function=SimpleNamespace(name="terminal", arguments='{"command":"echo $PPID"}'),
    )
    responses = [
        SimpleNamespace(
            content=None, finish_reason="tool_calls", reasoning=None, tool_calls=[call]
        ),
        SimpleNamespace(
            content="done", finish_reason="stop", reasoning=None, tool_calls=[]
        ),
    ]
    broker, adapter, child, completions, digest = _fixture(tmp_path, responses)
    child.tools = [
        {
            "type": "function",
            "function": {"name": "terminal", "parameters": {"type": "object"}},
        }
    ]
    child._delegate_frozen_dispatch_entries = {"terminal": object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=adapter.profile, task=adapter.task
    )
    broker_calls = []
    original_dispatch = adapter._dispatch

    def observed_dispatch(operation, body):
        if operation == "tool.execute":
            broker_calls.append(body)
        return original_dispatch(operation, body)

    monkeypatch.setattr(adapter, "_dispatch", observed_dispatch)
    worker_path = Path(__file__).parents[2] / "agent" / "subagent_worker_main.py"
    result = run_owned_process(
        ProcessRunSpec(
            executable=sys.executable,
            argv=(str(worker_path),),
            cwd=tmp_path,
            workspace=tmp_path,
            capability_secret=broker.reveal_secret_for_transport(),
            capability_id=broker.capability_id,
            launch_receipt_digest=digest,
            capability_fd_arg="--capability-fd",
            broker=adapter,
            timeout_seconds=5,
        )
    )
    assert result.state == "SUCCEEDED", (result.diagnostic, result.stderr)
    assert broker_calls == []
    tool_message = completions.kwargs["messages"][-1]
    assert tool_message["role"] == "tool"
    assert str(os.getpid()) not in str(tool_message["content"])


def test_nested_dispatch_records_only_digests_and_safe_public_attestation(
    tmp_path, monkeypatch
):
    _broker, adapter, child, _completions, _digest = _brokered_fixture(tmp_path)
    name = next(iter(child._delegate_frozen_dispatch_entries))
    frozen_entry = child._delegate_frozen_dispatch_entries[name]
    observed = []
    result_payload = {
        "ok": True,
        "status": "completed",
        "private_output": {"api_key": "sk-tes...alue"},
        "broker_attestation": _valid_broker_attestation(),
        "completion": {"result_hash": "b" * 64},
        "execution_receipt_hash": "c" * 64,
        "evidence": {"verified": True},
    }
    result_text = json.dumps(result_payload, indent=2)

    def dispatch(snapshot, tool_name, args, **kwargs):
        from agent.subagent_lifecycle import get_active_subagent_parent

        observed.append((
            snapshot,
            tool_name,
            args,
            get_active_subagent_parent(),
            kwargs,
        ))
        return result_text

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", dispatch
    )
    arguments = {
        **_valid_broker_arguments(tmp_path),
        "api_key": "«redacted:sk-…»",
        "task": "private delegated task",
    }
    assert adapter._dispatch(
        "tool.execute",
        {"id": "nested-1", "name": name, "arguments": arguments},
    ) == {"result": result_text}
    assert dict(observed[0][0]) == {name: frozen_entry}
    assert observed[0][1:4] == (name, arguments, child)

    claims = adapter.tool_execution_claims
    assert len(claims) == 1
    claim = claims[0]
    assert claim.tool_name == name
    assert (
        claim.arguments_sha256
        == hashlib.sha256(canonical_json(arguments).encode()).hexdigest()
    )
    assert (
        claim.result_sha256
        == hashlib.sha256(canonical_json(result_payload).encode()).hexdigest()
    )
    assert claim.public_attestation_json == canonical_json(
        result_payload["broker_attestation"]
    )
    assert set(claim.to_dict()) == {
        "sequence",
        "tool_name",
        "arguments_sha256",
        "result_sha256",
        "public_attestation_json",
        "launch_receipt_sha256",
        "tool_schema_sha256",
    }
    assert claim.sequence == 1
    assert claim.launch_receipt_sha256 == _digest
    assert claim.tool_schema_sha256 == process_integration.exact_tool_schema_digest(
        child.tools
    )
    persisted_claim = canonical_json(claim.to_dict())
    assert "private delegated task" not in persisted_claim
    assert "private-result-value" not in persisted_claim
    assert not hasattr(claim, "arguments_json")
    assert not hasattr(claim, "result_json")
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.tool_name = "changed"


def test_host_run_claim_binds_candidate_launch_schema_and_attestation(
    tmp_path, monkeypatch
):
    broker, old_adapter, child, _completions, digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_run"
    child.tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
    ]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="run attempt"
    )
    adapter.profile.execution_backend = "linux_strict"
    result_payload = _valid_run_result_payload()
    dispatch_kwargs = []

    def dispatch(*_args, **kwargs):
        dispatch_kwargs.append(kwargs)
        return result_payload

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        dispatch,
    )
    arguments = {
        "workspace": "/workspace",
        "experiment_id": "exp_1001",
        "attempt_n": 1,
        "timeout_seconds": 30,
    }
    host_effective_arguments = {**arguments, "workspace": str(tmp_path.resolve())}

    response = adapter._dispatch(
        "tool.execute",
        {"id": "run-1", "name": name, "arguments": arguments},
    )

    assert response == {"result": result_payload}
    claim = adapter.tool_execution_claims[0]
    assert claim.sequence == 1
    assert claim.tool_name == name
    assert (
        claim.arguments_sha256
        == hashlib.sha256(canonical_json(host_effective_arguments).encode()).hexdigest()
    )
    assert claim.launch_receipt_sha256 == digest
    assert claim.tool_schema_sha256 == process_integration.exact_tool_schema_digest(
        child.tools
    )
    assert claim.public_attestation_json == canonical_json(
        result_payload["broker_attestation"]
    )
    assert dispatch_kwargs[0]["cancellation_event"] is adapter._cancellation_event


def test_host_run_claim_rejects_attempt_execution_not_bound_to_host_receipt(
    tmp_path, monkeypatch
):
    broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_run"
    child.tools = [{"type": "function", "function": {"name": name}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="run attempt"
    )
    adapter.profile.execution_backend = "linux_strict"
    result_payload = _valid_run_result_payload()
    result_payload["execution_receipt"]["authority_sha256"] = "9" * 64
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result_payload,
    )

    with pytest.raises(ProcessIntegrationError, match="host execution receipt"):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "run-1",
                "name": name,
                "arguments": {
                    "workspace": "/workspace",
                    "experiment_id": "exp_1001",
                    "attempt_n": 1,
                    "timeout_seconds": 30,
                },
            },
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "receipt_hash",
        "result_hash",
        "cleanup_hash",
        "request_identity",
        "outcome",
    ],
)
def test_host_run_claim_rejects_unbound_exact_producer_payload(
    tmp_path, monkeypatch, mutation
):
    broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_run"
    child.tools = [{"type": "function", "function": {"name": name}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="run attempt"
    )
    adapter.profile.execution_backend = "linux_strict"
    result_payload = _valid_run_result_payload()
    if mutation == "missing_receipt":
        result_payload.pop("execution_receipt")
    elif mutation == "receipt_hash":
        result_payload["execution_receipt_hash"] = "9" * 64
        result_payload["broker_attestation"]["completion"]["execution_receipt_hash"] = (
            "9" * 64
        )
    elif mutation == "result_hash":
        result_payload["completion"]["result_hash"] = "9" * 64
        result_payload["broker_attestation"]["completion"]["result_hash"] = "9" * 64
    elif mutation == "cleanup_hash":
        result_payload["broker_attestation"]["attempt_execution"]["cleanup_sha256"] = (
            "9" * 64
        )
    elif mutation == "request_identity":
        result_payload["result"]["experiment_id"] = "exp_9999"
        digest = hashlib.sha256(
            canonical_json(result_payload["result"]).encode()
        ).hexdigest()
        result_payload["completion"]["result_hash"] = digest
        result_payload["broker_attestation"]["completion"]["result_hash"] = digest
    else:
        result_payload["broker_attestation"]["outcome"]["status"] = "evaluated"
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result_payload,
    )

    with pytest.raises(ProcessIntegrationError):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "run-1",
                "name": name,
                "arguments": {
                    "workspace": "/workspace",
                    "experiment_id": "exp_1001",
                    "attempt_n": 1,
                    "timeout_seconds": 30,
                },
            },
        )


def test_host_run_failure_latches_unresolved_side_effects(tmp_path, monkeypatch):
    broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_run"
    child.tools = [{"type": "function", "function": {"name": name}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="run attempt"
    )
    adapter.profile.execution_backend = "linux_strict"
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error_code": "evo_run_side_effects_unresolved",
            "error": "nested containment failed",
            "side_effects_unresolved": True,
        },
    )

    response = adapter._dispatch(
        "tool.execute",
        {
            "id": "run-1",
            "name": name,
            "arguments": {
                "workspace": "/workspace",
                "experiment_id": "exp_1001",
                "attempt_n": 1,
                "timeout_seconds": 30,
            },
        },
    )

    assert response["result"]["ok"] is False
    assert adapter.side_effects_unresolved is True
    assert adapter.tool_execution_claims == ()


def test_side_effecting_handler_exception_latches_unresolved_effects(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_run"
    _broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    child.tools = [{"type": "function", "function": {"name": name}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=old_adapter.broker,
        child=child,
        profile=old_adapter.profile,
        task="run attempt",
    )
    adapter.profile.execution_backend = "linux_strict"
    effects = []

    def crash_after_effect(*_args, **_kwargs):
        effects.append("workspace-mutated")
        raise RuntimeError("crash after effect")

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        crash_after_effect,
    )

    with pytest.raises(RuntimeError, match="crash after effect"):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "run-1",
                "name": name,
                "arguments": {
                    "workspace": "/workspace",
                    "experiment_id": "exp_1001",
                    "attempt_n": 1,
                    "timeout_seconds": 30,
                },
            },
        )
    assert effects == ["workspace-mutated"]
    assert adapter.side_effects_unresolved is True
    assert adapter.tool_execution_claims == ()


@pytest.mark.parametrize(
    ("prior_unresolved", "expected_unresolved"),
    [(False, False), (True, True)],
)
def test_contained_side_effecting_handler_failure_restores_unresolved_latch(
    tmp_path, monkeypatch, prior_unresolved, expected_unresolved
):
    name = "scaffolde_evo_run"
    _broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    child.tools = [{"type": "function", "function": {"name": name}}]
    child._delegate_frozen_dispatch_entries = {name: object()}
    adapter = ParentBrokerAdapter(
        broker=old_adapter.broker,
        child=child,
        profile=old_adapter.profile,
        task="run attempt",
    )
    adapter.profile.execution_backend = "linux_strict"
    adapter._side_effects_unresolved = prior_unresolved
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error_code": "evo_run_failed",
            "error": "nested attempt failed with conclusive cleanup",
        },
    )

    response = adapter._dispatch(
        "tool.execute",
        {
            "id": "run-1",
            "name": name,
            "arguments": {
                "workspace": "/workspace",
                "experiment_id": "exp_1001",
                "attempt_n": 1,
                "timeout_seconds": 30,
            },
        },
    )

    assert response["result"]["ok"] is False
    assert adapter.side_effects_unresolved is expected_unresolved
    assert adapter.tool_execution_claims == ()


def test_brokered_claim_rejects_attestation_not_bound_to_host_completion(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    attestation = _valid_broker_attestation()
    attestation["completion"] = {
        "result_hash": "f" * 64,
        "execution_receipt_hash": "c" * 64,
    }
    result_payload = {
        "ok": True,
        "broker_attestation": attestation,
        "completion": {"result_hash": "b" * 64},
        "execution_receipt_hash": "c" * 64,
        "evidence": {"verified": True},
    }
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result_payload,
    )

    with pytest.raises(ProcessIntegrationError, match="host completion"):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )


def test_brokered_dispatch_rejects_effective_tool_schema_drift(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, child, _completions, _digest = _brokered_fixture(tmp_path)
    child.tools[0]["function"]["parameters"]["properties"] = {
        "forged": {"type": "string"}
    }
    dispatched = []
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: dispatched.append(True),
    )

    with pytest.raises(
        ProcessIntegrationError, match="effective tool schema changed after launch"
    ):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )

    assert dispatched == []
    assert adapter.tool_execution_claims == ()


def test_brokered_scaffolde_tool_rejects_cross_workspace_dispatch(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    other_workspace = tmp_path.parent / "other-workspace"
    other_workspace.mkdir()
    dispatched = []
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: dispatched.append(True),
    )
    arguments = _valid_broker_arguments(tmp_path)
    arguments["workspace"] = str(other_workspace)

    with pytest.raises(
        ProcessIntegrationError, match="does not match the profile workspace"
    ):
        adapter._dispatch(
            "tool.execute",
            {"id": "nested-1", "name": name, "arguments": arguments},
        )

    assert dispatched == []
    assert adapter.tool_execution_claims == ()


def test_brokered_scaffolde_tool_maps_exact_linux_strict_workspace_alias(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    adapter.profile.execution_backend = "linux_strict"
    dispatched = []

    def dispatch(_snapshot, _name, arguments, **_kwargs):
        dispatched.append(arguments["workspace"])
        return {"ok": False, "error_code": "expected-stop", "error": "stop"}

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        dispatch,
    )
    arguments = _valid_broker_arguments(tmp_path)
    arguments["workspace"] = "/workspace"

    response = adapter._dispatch(
        "tool.execute",
        {"id": "nested-1", "name": name, "arguments": arguments},
    )

    assert dispatched == [str(tmp_path.resolve())]
    assert response["result"]["error_code"] == "expected-stop"


def test_failed_nested_dispatch_returns_bounded_error_without_claim(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error_code": "specialist_result_invalid",
            "error": "specialist returned malformed output",
            "unvalidated_execution_receipt": {"private": "must-not-cross"},
        },
    )

    assert adapter._dispatch(
        "tool.execute",
        {
            "id": "nested-1",
            "name": name,
            "arguments": _valid_broker_arguments(tmp_path),
        },
    ) == {
        "result": {
            "ok": False,
            "error_code": "specialist_result_invalid",
            "error": "specialist returned malformed output",
        }
    }
    assert adapter.tool_execution_claims == ()


def test_invalid_nested_arguments_reach_host_validation_without_a_claim(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    observed = []

    def reject_invalid_arguments(_snapshot, _name, arguments, **_kwargs):
        observed.append(arguments)
        return {
            "ok": False,
            "error_code": "attempt_binding_required",
            "error": "attempt_n must be explicit",
        }

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        reject_invalid_arguments,
    )
    arguments = {
        "workspace": str(tmp_path),
        "role": "verifier",
        "experiment_id": "exp_1001",
        "phase": "post",
    }

    assert adapter._dispatch(
        "tool.execute",
        {"id": "nested-1", "name": name, "arguments": arguments},
    ) == {
        "result": {
            "ok": False,
            "error_code": "attempt_binding_required",
            "error": "attempt_n must be explicit",
        }
    }
    assert observed == [arguments]
    assert adapter.tool_execution_claims == ()


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({}, "exactly one top-level broker_attestation mapping"),
        ('{"status":"completed"}', "exactly one top-level broker_attestation"),
        (
            {"broker_attestation": []},
            "exactly one top-level broker_attestation mapping",
        ),
        (
            '{"broker_attestation":{"status":"ok"},'
            '"broker_attestation":{"status":"duplicate"}}',
            "duplicate JSON key",
        ),
        ("not-json", "malformed JSON"),
        (object(), "JSON string or mapping"),
        (
            {"broker_attestation": {"api_key": "not-recordable"}},
            "does not match the Scaffolde schema",
        ),
        (
            {"broker_attestation": {"clientSecret": "not-recordable"}},
            "does not match the Scaffolde schema",
        ),
        (
            {"broker_attestation": {"messages": [{"role": "user", "content": "raw"}]}},
            "does not match the Scaffolde schema",
        ),
        (
            {"broker_attestation": {"evidence": object()}},
            "not canonical JSON",
        ),
        (
            {
                "broker_attestation": {
                    "evidence": "opaque-live-credential-value-1234567890"
                }
            },
            "does not match the Scaffolde schema",
        ),
    ],
)
def test_brokered_claim_rejects_missing_malformed_or_unsafe_attestation(
    tmp_path, monkeypatch, result, error
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(ProcessIntegrationError, match=error):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )
    assert adapter.tool_execution_claims == ()


def test_brokered_claim_rejects_oversized_arguments_before_execution(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(process_integration, "_MAX_BROKERED_TOOL_ARGUMENT_BYTES", 16)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", pytest.fail
    )

    with pytest.raises(ProcessIntegrationError, match="exceeds its byte bound"):
        adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": {"workspace": str(tmp_path), "task": "x" * 32},
            },
        )
    assert adapter.tool_execution_claims == ()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_brokered_claim_rejects_non_finite_json_numbers(tmp_path, monkeypatch, value):
    name = "scaffolde_evo_agent_dispatch"
    _broker, argument_adapter, _child, _completions, _digest = _brokered_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", pytest.fail
    )

    with pytest.raises(ProcessIntegrationError, match="not canonical JSON"):
        argument_adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": {"workspace": str(tmp_path), "score": value},
            },
        )
    assert argument_adapter.tool_execution_claims == ()

    _broker, result_adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: {
            "score": value,
            "broker_attestation": {"status": "completed"},
        },
    )

    with pytest.raises(ProcessIntegrationError, match="not canonical JSON"):
        result_adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )
    assert result_adapter.tool_execution_claims == ()


def test_brokered_claim_rejects_oversized_result_and_public_attestation(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    result = {
        "output": "x" * 64,
        "broker_attestation": {"status": "completed"},
    }
    _broker, result_adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(process_integration, "_MAX_BROKERED_TOOL_RESULT_BYTES", 32)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(ProcessIntegrationError, match="result exceeds its byte bound"):
        result_adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )
    assert result_adapter.tool_execution_claims == ()

    monkeypatch.setattr(
        process_integration, "_MAX_BROKERED_TOOL_RESULT_BYTES", 64 * 1024
    )
    monkeypatch.setattr(process_integration, "_MAX_BROKERED_TOOL_ATTESTATION_BYTES", 32)
    _broker, attestation_adapter, _child, _completions, _digest = _brokered_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: _valid_broker_result(),
    )

    with pytest.raises(
        ProcessIntegrationError, match="public attestation exceeds its byte bound"
    ):
        attestation_adapter._dispatch(
            "tool.execute",
            {
                "id": "nested-1",
                "name": name,
                "arguments": _valid_broker_arguments(tmp_path),
            },
        )
    assert attestation_adapter.tool_execution_claims == ()


def test_brokered_public_attestation_enforces_canonical_schema_byte_bound(
    tmp_path, monkeypatch
):
    name = "scaffolde_evo_agent_dispatch"
    body = {
        "id": "nested-1",
        "name": name,
        "arguments": _valid_broker_arguments(tmp_path),
    }
    exact_attestation = _valid_broker_attestation()
    exact_size = len(canonical_json(exact_attestation).encode())
    monkeypatch.setattr(
        process_integration, "_MAX_BROKERED_TOOL_ATTESTATION_BYTES", exact_size
    )
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: _valid_broker_result(attestation=exact_attestation),
    )

    adapter._dispatch("tool.execute", body)
    assert len(adapter.tool_execution_claims) == 1
    assert (
        len(adapter.tool_execution_claims[0].public_attestation_json.encode())
        == exact_size
    )

    monkeypatch.setattr(
        process_integration, "_MAX_BROKERED_TOOL_ATTESTATION_BYTES", exact_size - 1
    )
    _broker, bounded_adapter, _child, _completions, _digest = _brokered_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: _valid_broker_result(attestation=exact_attestation),
    )

    with pytest.raises(ProcessIntegrationError, match="exceeds its byte bound"):
        bounded_adapter._dispatch("tool.execute", body)
    assert bounded_adapter.tool_execution_claims == ()


def test_brokered_claims_enforce_aggregate_byte_bound(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: _valid_broker_result(summary="bounded"),
    )
    body = {
        "id": "nested-1",
        "name": name,
        "arguments": _valid_broker_arguments(tmp_path),
    }
    adapter._dispatch("tool.execute", body)
    first_claim = adapter.tool_execution_claims[0]
    first_size = len(canonical_json(first_claim.to_dict()).encode())
    monkeypatch.setattr(
        process_integration, "_MAX_BROKERED_TOOL_CLAIM_BYTES", first_size * 2 - 1
    )

    with pytest.raises(ProcessIntegrationError, match="aggregate byte bound"):
        adapter._dispatch("tool.execute", body)
    assert adapter.tool_execution_claims == (first_claim,)


def test_brokered_claim_count_reservation_is_race_safe(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(process_integration, "_MAX_BROKERED_TOOL_CLAIMS", 1)
    entered = threading.Event()
    release = threading.Event()
    outcomes = []

    def dispatch(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=1)
        return _valid_broker_result()

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", dispatch
    )
    body = {
        "id": "nested-1",
        "name": name,
        "arguments": _valid_broker_arguments(tmp_path),
    }

    def invoke():
        try:
            outcomes.append(adapter._dispatch("tool.execute", body))
        except Exception as exc:  # pragma: no cover - assertion reports below
            outcomes.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(timeout=1)
    try:
        with pytest.raises(ProcessIntegrationError, match="count exceeds"):
            adapter._dispatch("tool.execute", body)
    finally:
        release.set()
        thread.join(timeout=1)

    assert not thread.is_alive()
    assert outcomes == [{"result": _valid_broker_result()}]
    assert len(adapter.tool_execution_claims) == 1


def test_brokered_claim_aggregate_bound_is_race_safe(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    result = _valid_broker_result()
    body = {
        "id": "nested-1",
        "name": name,
        "arguments": _valid_broker_arguments(tmp_path),
    }
    _broker, probe, _child, _completions, _digest = _brokered_fixture(tmp_path)
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: result,
    )
    probe._dispatch("tool.execute", body)
    claim_size = len(canonical_json(probe.tool_execution_claims[0].to_dict()).encode())
    monkeypatch.setattr(process_integration, "_MAX_RESERVED_CLAIM_BYTES", claim_size)
    monkeypatch.setattr(
        process_integration, "_MAX_BROKERED_TOOL_CLAIM_BYTES", 3 + claim_size
    )

    _broker, adapter, _child, _completions, _digest = _brokered_fixture(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    outcomes = []
    dispatch_count = 0

    def dispatch(*_args, **_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        entered.set()
        assert release.wait(timeout=1)
        return result

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", dispatch
    )

    def invoke():
        try:
            adapter._dispatch("tool.execute", body)
            outcomes.append("recorded")
        except Exception as exc:  # pragma: no cover - assertion reports below
            outcomes.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(timeout=1)
    try:
        invoke()
    finally:
        release.set()
        thread.join(timeout=1)

    assert not thread.is_alive()
    assert outcomes.count("recorded") == 1
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(errors) == 1
    assert "aggregate byte bound" in str(errors[0])
    assert dispatch_count == 1
    assert len(adapter.tool_execution_claims) == 1


def test_worker_local_dispatch_cannot_broaden_to_ambient_registry(monkeypatch):
    snapshot = {"read_file": object()}
    monkeypatch.setattr(
        "tools.registry.registry.dispatch",
        lambda *_a, **_k: pytest.fail("ambient dispatch must not run"),
    )
    with pytest.raises(Exception, match="outside worker-local frozen authority"):
        _dispatch_local(snapshot, "terminal", {})


def test_strict_runtime_mounts_include_importable_worker_handler_modules():
    mounts = strict_worker_runtime_mounts()
    roots = {mount.source for mount in mounts}
    targets = {mount.target for mount in mounts}
    repository_root = Path(__file__).parents[2].resolve()
    assert repository_root in roots
    assert Path(sys.prefix).resolve() in roots
    assert Path(sys.base_prefix).resolve() in roots
    for system_root in (Path("/lib"), Path("/lib64"), Path("/usr/lib")):
        if system_root.exists():
            assert system_root in targets
    assert (repository_root / "tools" / "terminal_tool.py").is_file()
    assert (repository_root / "tools" / "file_tools.py").is_file()


def test_strict_evo_profile_mounts_declared_cli_and_system_utilities_only():
    mounts = strict_worker_runtime_mounts(expose_scaffolde_evo_run=True)
    sources = {mount.source for mount in mounts}
    targets = {mount.target for mount in mounts}

    evo = Path(shutil.which("evo") or "").resolve(strict=True)
    git = Path(shutil.which("git") or "").resolve(strict=True)
    assert any(evo == source or evo.is_relative_to(source) for source in sources)
    assert git in sources
    assert Path("/") not in targets
    assert Path("/usr/bin") not in targets
    environment = {"PATH": strict_worker_runtime_path(expose_scaffolde_evo_run=True)}
    assert (
        subprocess.run(
            ["evo", "--version"], env=environment, capture_output=True, check=False
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["git", "--version"], env=environment, capture_output=True, check=False
        ).returncode
        == 0
    )


def test_strict_evo_runtime_does_not_overlay_cli_inside_covered_prefix(tmp_path):
    from agent.subagent_process_runner import RuntimeMount

    prefix = tmp_path / "usr-local"
    command = prefix / "bin" / "evo"
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mount = RuntimeMount(source=prefix, target=prefix)

    assert process_integration._runtime_target_is_covered({str(prefix): mount}, command)


def test_parent_dispatch_error_returns_authenticated_rejection(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    call = SimpleNamespace(
        id="nested-error",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(_valid_broker_arguments(tmp_path)),
        ),
    )
    responses = [
        SimpleNamespace(
            content=None, finish_reason="tool_calls", reasoning=None, tool_calls=[call]
        )
    ]
    broker, old_adapter, child, _completions, digest = _fixture(tmp_path, responses)
    child.tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
    ]
    child._delegate_frozen_dispatch_entries = {name: object()}
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="nested task"
    )
    host, worker = socket.socketpair()
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: adapter.serve(host, root_pid=123, stop_requested=stop),
        daemon=True,
    )
    thread.start()
    with pytest.raises(
        BrokerFrameError,
        match="broker rejected tool.execute operation: RuntimeError",
    ):
        run_worker_loop(
            worker,
            broker.reveal_secret_for_transport(),
            capability_id=broker.capability_id,
            launch_receipt_digest=digest,
        )
    stop.set()
    worker.shutdown(socket.SHUT_RDWR)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert adapter.tool_execution_claims == ()
