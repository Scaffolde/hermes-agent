import hashlib
import socket
import sys
import threading
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.subagent_broker_protocol import BrokerGrant, SubagentBroker
from agent.subagent_process_integration import (
    ParentBrokerAdapter,
    ProcessIntegrationError,
    strict_worker_runtime_mounts,
)
from agent.subagent_worker_main import (
    BrokerFrameError,
    _dispatch_local,
    run_worker_loop,
)
from agent.subagent_process_runner import ProcessRunSpec, run_owned_process


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
        protocol_text="strict protocol",
        max_process_iterations=4,
        workspace_root=str(tmp_path),
    )
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=profile, task="inspect task"
    )
    return broker, adapter, child, completions, launch_digest


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
            "tool.execute", {"id": "1", "name": "read_file", "arguments": {}}
        )


def test_worker_loop_executes_exact_tool_call_then_finishes(tmp_path, monkeypatch):
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
    local_calls = []
    monkeypatch.setattr(
        "agent.subagent_worker_main._dispatch_local",
        lambda snapshot, name, arguments: (
            local_calls.append((snapshot, name, arguments, __import__("os").getpid()))
            or "file contents"
        ),
    )
    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot",
        lambda *_args, **_kwargs: pytest.fail("parent registry dispatch must not run"),
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
    assert local_calls and local_calls[0][1] == "read_file"
    assert local_calls[0][3] == __import__("os").getpid()
    stop.set()
    worker.shutdown(socket.SHUT_RDWR)
    thread.join(timeout=1)
    worker.close()
    host.close()


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


def test_nested_dispatch_is_parent_brokered_from_frozen_entry(tmp_path, monkeypatch):
    broker, old_adapter, child, _completions, _digest = _fixture(tmp_path, [])
    name = "scaffolde_evo_agent_dispatch"
    child.tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
    ]
    child._delegate_frozen_dispatch_entries = {name: object()}
    frozen_entry = object()
    child._delegate_frozen_dispatch_entries = {name: frozen_entry}
    observed = []

    def dispatch(snapshot, tool_name, args, **kwargs):
        from agent.subagent_lifecycle import get_active_subagent_parent

        observed.append((
            snapshot,
            tool_name,
            args,
            get_active_subagent_parent(),
            kwargs,
        ))
        return "nested"

    monkeypatch.setattr(
        "agent.subagent_process_integration.registry.dispatch_snapshot", dispatch
    )
    adapter = ParentBrokerAdapter(
        broker=broker, child=child, profile=old_adapter.profile, task="nested task"
    )
    assert adapter._dispatch(
        "tool.execute",
        {"id": "nested-1", "name": name, "arguments": {"agent": "reviewer"}},
    ) == {"result": "nested"}
    assert dict(observed[0][0]) == {name: frozen_entry}
    assert observed[0][1:4] == (name, {"agent": "reviewer"}, child)


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


def test_parent_dispatch_error_returns_authenticated_rejection(tmp_path, monkeypatch):
    name = "scaffolde_evo_agent_dispatch"
    call = SimpleNamespace(
        id="nested-error",
        function=SimpleNamespace(name=name, arguments='{"agent":"reviewer"}'),
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
