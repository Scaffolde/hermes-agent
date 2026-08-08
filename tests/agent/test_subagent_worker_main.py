from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct

import pytest

import agent.subagent_worker_main as worker_main_module
from agent.subagent_worker_main import (
    BrokerFrameError,
    _enter_cgroup_and_exec,
    read_authenticated_frame,
    read_capability_secret,
    send_authenticated_frame,
    send_capability_secret,
    serve_one,
)


SECRET = b"s" * 32


def test_capability_secret_round_trip_uses_socket_only():
    host, worker = socket.socketpair()
    try:
        send_capability_secret(host, SECRET)
        assert read_capability_secret(worker) == SECRET
    finally:
        host.close()
        worker.close()


def test_authenticated_frame_round_trip_and_canonical_mac():
    host, worker = socket.socketpair()
    body = {"operation": "ping", "sequence": 1, "body": {"z": 2, "a": 1}}
    try:
        send_authenticated_frame(host, body, SECRET)
        assert read_authenticated_frame(worker, SECRET) == body
    finally:
        host.close()
        worker.close()


def test_authenticated_frame_rejects_tampered_mac():
    host, worker = socket.socketpair()
    body = {"operation": "ping"}
    body_bytes = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    wrong_mac = hmac.new(b"x" * 32, body_bytes, hashlib.sha256).hexdigest()
    outer = json.dumps(
        {"body": body, "mac": wrong_mac},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        host.sendall(struct.pack("!I", len(outer)) + outer)
        with pytest.raises(BrokerFrameError, match="authentication"):
            read_authenticated_frame(worker, SECRET)
    finally:
        host.close()
        worker.close()


def test_frame_length_is_rejected_before_payload_read():
    host, worker = socket.socketpair()
    try:
        host.sendall(struct.pack("!I", 4097))
        with pytest.raises(BrokerFrameError, match="length"):
            read_authenticated_frame(worker, SECRET, max_frame_bytes=4096)
    finally:
        host.close()
        worker.close()


def test_truncated_frame_is_rejected():
    host, worker = socket.socketpair()
    try:
        host.sendall(struct.pack("!I", 10) + b"tiny")
        host.shutdown(socket.SHUT_WR)
        with pytest.raises(BrokerFrameError, match="truncated"):
            read_authenticated_frame(worker, SECRET)
    finally:
        host.close()
        worker.close()


def test_serve_one_never_passes_secret_to_handler():
    host, worker = socket.socketpair()
    seen = []

    def handler(request):
        seen.append(request)
        return {"ok": True}

    try:
        send_capability_secret(host, SECRET)
        send_authenticated_frame(host, {"operation": "run", "goal": "safe"}, SECRET)
        serve_one(worker, handler)
        assert read_authenticated_frame(host, SECRET) == {"ok": True}
        assert seen == [{"operation": "run", "goal": "safe"}]
    finally:
        host.close()
        worker.close()


def test_cgroup_launcher_enters_before_exec_and_preserves_argv(monkeypatch):
    read_fd, write_fd = os.pipe()
    captured = {}

    def fake_execvpe(executable, argv, env):
        captured.update(executable=executable, argv=argv, env=env)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(os, "execvpe", fake_execvpe)
    try:
        with pytest.raises(RuntimeError, match="intercepted"):
            _enter_cgroup_and_exec(
                write_fd,
                ["/usr/bin/bwrap", "--", "/runtime/python", "worker.py"],
                {"LANG": "C.UTF-8"},
            )
        written = os.read(read_fd, 64)
    finally:
        os.close(read_fd)

    assert written == str(os.getpid()).encode("ascii")
    assert captured == {
        "executable": "/usr/bin/bwrap",
        "argv": ["/usr/bin/bwrap", "--", "/runtime/python", "worker.py"],
        "env": {"LANG": "C.UTF-8"},
    }


def test_cgroup_launcher_parser_preserves_bwrap_separator(monkeypatch):
    captured = {}

    def fake_enter(fd, argv, environment):
        captured.update(fd=fd, argv=argv, environment=environment)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(worker_main_module, "_enter_cgroup_and_exec", fake_enter)
    with pytest.raises(RuntimeError, match="intercepted"):
        worker_main_module.main([
            "--enter-cgroup-fd",
            "7",
            "--exec",
            "/usr/bin/bwrap",
            "--chdir",
            "/workspace",
            "--",
            "/runtime/python",
            "worker.py",
            "--capability-fd",
            "8",
        ])

    assert captured["fd"] == 7
    assert captured["argv"] == [
        "/usr/bin/bwrap",
        "--chdir",
        "/workspace",
        "--",
        "/runtime/python",
        "worker.py",
        "--capability-fd",
        "8",
    ]


def test_iteration_exhaustion_reports_safe_bounded_tool_trace(monkeypatch):
    tool = {
        "type": "function",
        "function": {
            "name": "scaffolde_evo_run",
            "description": "Run one attempt",
            "parameters": {"type": "object"},
        },
    }
    session = {
        "protocol": "protocol",
        "task": "task",
        "tools": [tool],
        "tool_schema_digest": worker_main_module.exact_tool_schema_digest([tool]),
        "local_tool_names": [],
        "brokered_tool_names": ["scaffolde_evo_run"],
        "model": "test-model",
        "max_iterations": 2,
    }
    model_calls = 0

    def fake_broker_call(_channel, _secret, _signer, operation, _body):
        nonlocal model_calls
        if operation == "session.start":
            return session
        if operation == "model.complete":
            model_calls += 1
            return {
                "finish_reason": "tool_calls",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{model_calls}",
                        "name": "scaffolde_evo_run",
                        "arguments": {
                            "workspace": "/secret/workspace",
                            "token": "DO_NOT_LEAK_ARGUMENT",
                        },
                    }
                ],
            }
        assert operation == "tool.execute"
        return {
            "result": json.dumps({
                "ok": False,
                "error_code": "evo_run_failed",
                "error": "DO_NOT_LEAK_RESULT",
            })
        }

    monkeypatch.setattr(worker_main_module, "_broker_call", fake_broker_call)

    with socket.socket() as channel:
        with pytest.raises(BrokerFrameError) as caught:
            worker_main_module.run_worker_loop(
                channel,
                SECRET,
                capability_id="capability",
                launch_receipt_digest="a" * 64,
            )

    message = str(caught.value)
    assert "worker iteration bound exhausted" in message
    assert '"iteration":2' in message
    assert '"tool_name":"scaffolde_evo_run"' in message
    assert '"error_code":"evo_run_failed"' in message
    assert "DO_NOT_LEAK_ARGUMENT" not in message
    assert "/secret/workspace" not in message
    assert "DO_NOT_LEAK_RESULT" not in message
