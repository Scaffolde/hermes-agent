"""Minimal worker-side bootstrap for owned Hermes subagent processes.

The capability secret crosses only an inherited Unix socket.  It is consumed
before authenticated, length-prefixed broker frames are accepted and is never
placed in argv, the environment, diagnostics, or returned payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import struct
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn, Protocol

# The owned worker is launched by absolute script path with a scrubbed
# environment. Pin its repository import root from that path; never consult an
# ambient PYTHONPATH or plugin directory.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from agent.subagent_tool_boundary import (
    LOCAL_EVO_TOOL_NAMES,
    classify_evo_tools,
    exact_tool_schema_digest,
)

MAX_BROKER_FRAME_BYTES = 1_048_576
MIN_CAPABILITY_SECRET_BYTES = 32
MAX_CAPABILITY_SECRET_BYTES = 128
MAX_WORKER_RESULT_BYTES = 1_048_576
_SECRET_MAGIC = b"HSEC1"
_LENGTH = struct.Struct("!I")
_SECRET_LENGTH = struct.Struct("!H")
_BROKER_PROTOCOL_VERSION = 1


class BrokerFrameError(ValueError):
    """A worker broker frame is malformed, oversized, or unauthenticated."""


class WorkerRequestHandler(Protocol):
    """Narrow integration seam for the serial lifecycle lane."""

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class _EnvelopeSigner:
    """Worker-local Lane A signer; keeps the worker free of package imports."""

    def __init__(self, capability_id: str, secret: bytes, launch_digest: str) -> None:
        self.capability_id = capability_id
        self.secret = secret
        self.launch_digest = launch_digest
        self.sequence = 0

    def sign(self, operation: str, body: Mapping[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        canonical_body = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        authority = {
            "protocol_version": _BROKER_PROTOCOL_VERSION,
            "capability_id": self.capability_id,
            "launch_receipt_digest": self.launch_digest,
            "sequence": self.sequence,
            "operation": operation,
            "body_digest": hashlib.sha256(canonical_body.encode("utf-8")).hexdigest(),
        }
        mac_payload = json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return {
            **authority,
            "body": dict(body),
            "mac": hmac.new(self.secret, mac_payload, hashlib.sha256).hexdigest(),
        }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BrokerFrameError("broker frame body is not canonical JSON") from exc


def _validate_secret(secret: bytes) -> bytes:
    if not isinstance(secret, bytes):
        raise BrokerFrameError("capability secret must be bytes")
    if not MIN_CAPABILITY_SECRET_BYTES <= len(secret) <= MAX_CAPABILITY_SECRET_BYTES:
        raise BrokerFrameError("capability secret length is outside the allowed bounds")
    return secret


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise BrokerFrameError("broker frame is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_capability_secret(channel: socket.socket, secret: bytes) -> None:
    """Bootstrap ``secret`` over an already inherited descriptor."""

    validated = _validate_secret(secret)
    channel.sendall(_SECRET_MAGIC + _SECRET_LENGTH.pack(len(validated)) + validated)


def read_capability_secret(channel: socket.socket) -> bytes:
    """Read the one-time capability bootstrap without rendering its value."""

    magic = _recv_exact(channel, len(_SECRET_MAGIC))
    if not hmac.compare_digest(magic, _SECRET_MAGIC):
        raise BrokerFrameError("invalid capability bootstrap")
    (length,) = _SECRET_LENGTH.unpack(_recv_exact(channel, _SECRET_LENGTH.size))
    if not MIN_CAPABILITY_SECRET_BYTES <= length <= MAX_CAPABILITY_SECRET_BYTES:
        raise BrokerFrameError("capability secret length is outside the allowed bounds")
    return _recv_exact(channel, length)


def send_worker_bootstrap(
    channel: socket.socket,
    secret: bytes,
    *,
    capability_id: str,
    launch_receipt_digest: str,
) -> None:
    send_capability_secret(channel, secret)
    send_authenticated_frame(
        channel,
        {
            "capability_id": capability_id,
            "launch_receipt_digest": launch_receipt_digest,
        },
        secret,
    )


def read_worker_bootstrap(channel: socket.socket) -> tuple[bytes, str, str]:
    secret = read_capability_secret(channel)
    authority = read_authenticated_frame(channel, secret)
    if set(authority) != {"capability_id", "launch_receipt_digest"}:
        raise BrokerFrameError("worker bootstrap authority has an invalid shape")
    capability_id = authority["capability_id"]
    launch_digest = authority["launch_receipt_digest"]
    if not isinstance(capability_id, str) or not isinstance(launch_digest, str):
        raise BrokerFrameError("worker bootstrap authority has invalid types")
    return secret, capability_id, launch_digest


def send_authenticated_frame(
    channel: socket.socket,
    body: Mapping[str, Any],
    secret: bytes,
    *,
    max_frame_bytes: int = MAX_BROKER_FRAME_BYTES,
) -> None:
    """Send one canonical JSON body authenticated with HMAC-SHA256."""

    validated = _validate_secret(secret)
    body_bytes = _canonical_json(body)
    mac = hmac.new(validated, body_bytes, hashlib.sha256).hexdigest()
    outer = _canonical_json({"body": dict(body), "mac": mac})
    if not 0 < len(outer) <= max_frame_bytes:
        raise BrokerFrameError("broker frame length exceeds the configured bound")
    channel.sendall(_LENGTH.pack(len(outer)) + outer)


def read_authenticated_frame(
    channel: socket.socket,
    secret: bytes,
    *,
    max_frame_bytes: int = MAX_BROKER_FRAME_BYTES,
) -> dict[str, Any]:
    """Receive and authenticate one bounded canonical JSON broker frame."""

    validated = _validate_secret(secret)
    (length,) = _LENGTH.unpack(_recv_exact(channel, _LENGTH.size))
    if not 0 < length <= max_frame_bytes:
        raise BrokerFrameError("broker frame length exceeds the configured bound")
    raw = _recv_exact(channel, length)
    try:
        outer = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrokerFrameError("broker frame is not valid UTF-8 JSON") from exc
    if not isinstance(outer, dict) or set(outer) != {"body", "mac"}:
        raise BrokerFrameError("broker frame envelope has an invalid shape")
    body = outer["body"]
    mac = outer["mac"]
    if not isinstance(body, dict) or not isinstance(mac, str):
        raise BrokerFrameError("broker frame envelope has invalid field types")
    expected = hmac.new(validated, _canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        raise BrokerFrameError("broker frame authentication failed")
    return body


def serve_one(channel: socket.socket, handler: WorkerRequestHandler) -> None:
    """Authenticate one request and response around a secret-blind handler."""

    secret = read_capability_secret(channel)
    request = read_authenticated_frame(channel, secret)
    response = handler(request)
    if not isinstance(response, Mapping):
        raise BrokerFrameError("worker handler response must be a mapping")
    send_authenticated_frame(channel, response, secret)


def _broker_call(
    channel: socket.socket,
    secret: bytes,
    signer: _EnvelopeSigner,
    operation: str,
    body: Mapping[str, Any],
) -> Mapping[str, Any]:
    send_authenticated_frame(channel, signer.sign(operation, dict(body)), secret)
    response = read_authenticated_frame(channel, secret)
    if set(response) != {"sequence", "ok", "body"}:
        raise BrokerFrameError("broker response has an invalid shape")
    if response["sequence"] != signer.sequence:
        raise BrokerFrameError("broker response is out of sequence")
    if not isinstance(response["body"], Mapping):
        raise BrokerFrameError("broker response body must be an object")
    if not response["ok"]:
        error_class = response["body"].get("error")
        if not isinstance(error_class, str) or not error_class:
            error_class = "unknown-error"
        raise BrokerFrameError(
            f"broker rejected {operation} operation: {error_class[:1000]}"
        )
    return response["body"]


def _schema_map(tools: list[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
            raise BrokerFrameError("session tool schema has an invalid shape")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if (
            tool.get("type") != "function"
            or not isinstance(name, str)
            or name in result
        ):
            raise BrokerFrameError("session tool schema has an invalid name")
        result[name] = function
    return result


def _freeze_local_handlers(
    names: frozenset[str], schemas: Mapping[str, dict[str, Any]]
) -> Mapping[str, Any]:
    if not names <= LOCAL_EVO_TOOL_NAMES:
        raise BrokerFrameError("worker local authority exceeds Evo v1 handlers")
    if names:
        import tools.file_tools  # noqa: F401
        import tools.terminal_tool  # noqa: F401
    from tools.registry import registry

    snapshot = registry.snapshot_dispatch_entries(
        set(names), effective_schemas={name: schemas[name] for name in names}
    )
    if set(snapshot) != set(names):
        raise BrokerFrameError("worker local handler is unavailable")
    return snapshot


def _dispatch_local(
    snapshot: Mapping[str, Any], name: str, arguments: Mapping[str, Any]
) -> str | dict[str, Any]:
    if name not in snapshot:
        raise BrokerFrameError("tool is outside worker-local frozen authority")
    from tools.registry import registry

    return registry.dispatch_snapshot(
        snapshot,
        name,
        dict(arguments),
        task_id=f"evo-worker-{os.getpid()}",
        enabled_tools=set(snapshot),
    )


def run_worker_loop(
    channel: socket.socket,
    secret: bytes,
    *,
    capability_id: str,
    launch_receipt_digest: str,
) -> dict[str, Any]:
    """Run the bounded credential-free worker conversation loop."""
    signer = _EnvelopeSigner(capability_id, secret, launch_receipt_digest)
    session = _broker_call(channel, secret, signer, "session.start", {})
    required = {
        "protocol",
        "task",
        "tools",
        "tool_schema_digest",
        "local_tool_names",
        "brokered_tool_names",
        "model",
        "max_iterations",
    }
    if set(session) != required:
        raise BrokerFrameError("session.start response has an invalid shape")
    if not isinstance(session["protocol"], str) or not isinstance(session["task"], str):
        raise BrokerFrameError("session text must be strings")
    if not isinstance(session["tools"], list) or not isinstance(session["model"], str):
        raise BrokerFrameError("session tools/model are malformed")
    schemas = _schema_map(session["tools"])
    digest = session["tool_schema_digest"]
    if not isinstance(digest, str) or digest != exact_tool_schema_digest(
        session["tools"]
    ):
        raise BrokerFrameError("session tool schema digest mismatch")
    local_raw = session["local_tool_names"]
    brokered_raw = session["brokered_tool_names"]
    if not isinstance(local_raw, list) or not all(
        isinstance(v, str) for v in local_raw
    ):
        raise BrokerFrameError("session local tool classification is malformed")
    if not isinstance(brokered_raw, list) or not all(
        isinstance(v, str) for v in brokered_raw
    ):
        raise BrokerFrameError("session brokered tool classification is malformed")
    local_names, brokered_names = classify_evo_tools(schemas)
    if local_raw != sorted(local_names) or brokered_raw != sorted(brokered_names):
        raise BrokerFrameError(
            "session tool classification does not match exact schemas"
        )
    local_handlers = _freeze_local_handlers(local_names, schemas)
    limit = session["max_iterations"]
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise BrokerFrameError("max_iterations is outside the worker bound")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": session["protocol"]},
        {"role": "user", "content": session["task"]},
    ]
    seen_call_ids: set[str] = set()
    for iteration in range(limit):
        completion = _broker_call(
            channel,
            secret,
            signer,
            "model.complete",
            {"messages": messages},
        )
        finish = completion.get("finish_reason")
        content = completion.get("content")
        tool_calls = completion.get("tool_calls", [])
        if finish not in {"stop", "tool_calls"} or not isinstance(tool_calls, list):
            raise BrokerFrameError("model returned an unsupported finish state")
        if content is not None and not isinstance(content, str):
            raise BrokerFrameError("model content must be text or null")
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            wire_tool_calls = []
            for call in tool_calls:
                if not isinstance(call, Mapping) or set(call) != {
                    "id",
                    "name",
                    "arguments",
                }:
                    raise BrokerFrameError("tool call has an invalid shape")
                call_id, name, arguments = call["id"], call["name"], call["arguments"]
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or call_id in seen_call_ids
                ):
                    raise BrokerFrameError("tool call id is missing or duplicated")
                if not isinstance(name, str) or not isinstance(arguments, Mapping):
                    raise BrokerFrameError("tool call name/arguments are malformed")
                seen_call_ids.add(call_id)
                wire_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            arguments,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    },
                })
            assistant["tool_calls"] = wire_tool_calls
        messages.append(assistant)
        if finish == "stop":
            if tool_calls:
                raise BrokerFrameError("stop response must not contain tool calls")
            return {"summary": content or "", "iterations": iteration + 1}
        if not tool_calls:
            raise BrokerFrameError("tool_calls finish requires tool calls")
        for call in tool_calls:
            call_id, name, arguments = call["id"], call["name"], call["arguments"]
            if name in local_names:
                result = {"result": _dispatch_local(local_handlers, name, arguments)}
            elif name in brokered_names:
                result = _broker_call(
                    channel,
                    secret,
                    signer,
                    "tool.execute",
                    {"id": call_id, "name": name, "arguments": dict(arguments)},
                )
            else:
                raise BrokerFrameError("model requested a tool outside exact authority")
            if set(result) != {"result"} or not isinstance(
                result["result"], (str, list, dict)
            ):
                raise BrokerFrameError("tool result is malformed")
            tool_content = result["result"]
            if not isinstance(tool_content, str):
                tool_content = json.dumps(
                    tool_content,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_content,
            })
    raise BrokerFrameError("worker iteration bound exhausted")


def _default_handler(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Readiness-only default; real execution is supplied by serial integration."""

    if request == {"operation": "ping"}:
        return {"ok": True, "operation": "pong"}
    return {"ok": False, "error": "unsupported-operation"}


def _write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("cgroup.procs write made no progress")
        view = view[written:]


def _enter_cgroup_and_exec(
    cgroup_procs_fd: int,
    argv: list[str],
    environment: Mapping[str, str],
) -> NoReturn:
    """Enter a prepared cgroup before exec, so every descendant is owned.

    The launcher and bwrap use the same PID because this ends in ``execvpe``.
    No worker instruction can run before the kernel has accepted the PID in
    the dedicated cgroup.
    """

    if cgroup_procs_fd < 0:
        raise ValueError("cgroup_procs_fd must be non-negative")
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise ValueError("exec argv must contain safe strings")
    if not os.path.isabs(argv[0]):
        raise ValueError("exec executable must be absolute")
    try:
        _write_all(cgroup_procs_fd, str(os.getpid()).encode("ascii"))
    finally:
        os.close(cgroup_procs_fd)
    os.execvpe(argv[0], argv, dict(environment))
    raise AssertionError("os.execvpe unexpectedly returned")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capability-fd", type=int)
    parser.add_argument("--enter-cgroup-fd", type=int)
    parser.add_argument("--exec", dest="exec_mode", action="store_true")
    parser.add_argument("exec_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.enter_cgroup_fd is not None:
        if not args.exec_mode or not args.exec_argv:
            raise SystemExit("--enter-cgroup-fd requires --exec argv")
        _enter_cgroup_and_exec(args.enter_cgroup_fd, args.exec_argv, os.environ)
    if args.capability_fd is None or args.exec_mode or args.exec_argv:
        raise SystemExit("exactly one worker mode is required")
    with socket.socket(fileno=args.capability_fd) as channel:
        secret, capability_id, launch_digest = read_worker_bootstrap(channel)
        if capability_id == "runner-capability" and launch_digest == "0" * 64:
            request = read_authenticated_frame(channel, secret)
            send_authenticated_frame(channel, _default_handler(request), secret)
            return 0
        result = run_worker_loop(
            channel,
            secret,
            capability_id=capability_id,
            launch_receipt_digest=launch_digest,
        )
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_WORKER_RESULT_BYTES:
        raise SystemExit("worker result exceeds output bound")
    os.write(1, encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
