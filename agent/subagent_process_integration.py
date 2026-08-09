"""Host-side adapter for strict profiled subprocess execution."""

from __future__ import annotations

import dataclasses
import json
import socket
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent.subagent_broker_protocol import SubagentBroker, canonical_json
from agent.subagent_worker_main import (
    BrokerFrameError,
    read_authenticated_frame,
    send_authenticated_frame,
)
from agent.subagent_lifecycle import bind_subagent_parent
from agent.subagent_tool_boundary import (
    EvoToolBoundaryError,
    classify_evo_tools,
    exact_tool_schema_digest,
)
from tools.registry import FrozenToolDispatchEntry, ToolRegistry, registry


class ProcessIntegrationError(RuntimeError):
    pass


def _bounded_diagnostic_text(value: object, *, limit: int) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(value)
    )
    return " ".join(printable.split())[:limit]


def strict_worker_runtime_mounts() -> tuple[Any, ...]:
    """Return the minimal declared runtime needed by worker-local handlers."""
    from agent.subagent_process_runner import RuntimeMount

    python_prefix = Path(sys.prefix).resolve(strict=True)
    python_base_prefix = Path(sys.base_prefix).resolve(strict=True)
    repository_root = Path(__file__).parents[1].resolve(strict=True)
    mounts = {
        str(python_prefix): RuntimeMount(source=python_prefix, target=python_prefix),
        str(python_base_prefix): RuntimeMount(
            source=python_base_prefix, target=python_base_prefix
        ),
        str(repository_root): RuntimeMount(
            source=repository_root, target=repository_root
        ),
    }
    for target in (Path("/lib"), Path("/lib64"), Path("/usr/lib")):
        if target.exists():
            mounts.setdefault(
                str(target),
                RuntimeMount(source=target.resolve(strict=True), target=target),
            )
    return tuple(mounts.values())


def launch_receipt_digest(receipt: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(receipt.to_dict()).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    raise ProcessIntegrationError(f"unsupported non-JSON value: {type(value).__name__}")


def _tool_calls(normalized: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in getattr(normalized, "tool_calls", None) or []:
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id or call_id in seen:
            raise ProcessIntegrationError(
                "model emitted missing or duplicate tool call id"
            )
        if not isinstance(name, str) or not name:
            raise ProcessIntegrationError("model emitted an invalid tool name")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ProcessIntegrationError(
                    "model emitted invalid tool arguments"
                ) from exc
        if not isinstance(arguments, Mapping):
            raise ProcessIntegrationError("model tool arguments must be a JSON object")
        seen.add(call_id)
        result.append({"id": call_id, "name": name, "arguments": _json_safe(arguments)})
    return result


class ParentBrokerAdapter:
    """Authenticate worker frames and perform only three host-owned operations."""

    def __init__(
        self, *, broker: SubagentBroker, child: Any, profile: Any, task: str
    ) -> None:
        self.broker = broker
        self.child = child
        self.profile = profile
        self.task = task
        self._secret = broker.reveal_secret_for_transport()
        try:
            self.local_tool_names, self.brokered_tool_names = classify_evo_tools(
                child._delegate_frozen_dispatch_entries
            )
        except EvoToolBoundaryError as exc:
            raise ProcessIntegrationError(str(exc)) from exc
        self._brokered_dispatch_entries: Mapping[str, FrozenToolDispatchEntry] = (
            MappingProxyType({
                name: child._delegate_frozen_dispatch_entries[name]
                for name in self.brokered_tool_names
            })
        )
        if set(self._brokered_dispatch_entries) != set(self.brokered_tool_names):
            raise ProcessIntegrationError("host-brokered frozen handler is unavailable")

    def serve(
        self,
        channel: socket.socket,
        *,
        root_pid: int,
        stop_requested: threading.Event,
    ) -> None:
        del root_pid
        while not stop_requested.is_set():
            sequence: int | None = None
            request = None
            try:
                envelope = read_authenticated_frame(channel, self._secret)
            except BrokerFrameError as exc:
                if "truncated" in str(exc):
                    return
                raise
            try:
                request = self.broker.validate(envelope)
                sequence = request.sequence
                body = self._dispatch(request.operation, request.body)
                response = {"sequence": sequence, "ok": True, "body": body}
            except Exception as exc:
                if sequence is None:
                    raise
                error = type(exc).__name__
                error_body = getattr(exc, "body", None)
                if isinstance(error_body, Mapping):
                    safe_parts = []
                    for key in ("code", "param", "message"):
                        value = error_body.get(key)
                        if isinstance(value, (str, int, float)) and value != "":
                            safe_value = _bounded_diagnostic_text(value, limit=300)
                            safe_parts.append(f"{key}={safe_value}")
                    if safe_parts:
                        error = f"{error}: {'; '.join(safe_parts)}"
                if (
                    request is not None
                    and request.operation == "model.complete"
                    and isinstance(request.body, Mapping)
                ):
                    raw_messages = request.body.get("messages")
                    if isinstance(raw_messages, list):
                        message_shapes = []
                        for message in raw_messages:
                            if not isinstance(message, Mapping):
                                message_shapes.append("invalid")
                                continue
                            content = message.get("content")
                            message_shapes.append(
                                f"{message.get('role')}:{type(content).__name__}:"
                                f"{len(content) if isinstance(content, (str, list)) else -1}"
                            )
                        error = (
                            f"{error}; message_shapes={','.join(message_shapes)[:500]}"
                        )
                response = {
                    "sequence": sequence,
                    "ok": False,
                    "body": {"error": error},
                }
            send_authenticated_frame(channel, response, self._secret)

    def _dispatch(self, operation: str, body: Any) -> Mapping[str, Any]:
        if not isinstance(body, Mapping):
            raise ProcessIntegrationError("broker operation body must be an object")
        if operation == "session.start":
            if body:
                raise ProcessIntegrationError("session.start body must be empty")
            return {
                "protocol": self.profile.protocol_text,
                "task": self.task,
                "tools": _json_safe(self.child.tools),
                "tool_schema_digest": exact_tool_schema_digest(self.child.tools),
                "local_tool_names": sorted(self.local_tool_names),
                "brokered_tool_names": sorted(self.brokered_tool_names),
                "model": str(self.child.model or ""),
                "max_iterations": self.profile.max_process_iterations,
            }
        if operation == "model.complete":
            if set(body) != {"messages"} or not isinstance(body["messages"], list):
                raise ProcessIntegrationError("model.complete body is malformed")
            kwargs = self.child._build_api_kwargs(
                body["messages"], tools_for_api=self.child.tools
            )
            kwargs["stream"] = False
            kwargs.pop("stream_options", None)
            response = self.child.client.chat.completions.create(**kwargs)
            normalized = self.child._get_transport().normalize_response(response)
            finish = getattr(normalized, "finish_reason", None)
            if finish not in {"stop", "tool_calls"}:
                raise ProcessIntegrationError("model returned unsupported finish state")
            content = getattr(normalized, "content", None)
            if content is not None and not isinstance(content, str):
                content = json.dumps(_json_safe(content), sort_keys=True)
            usage = getattr(normalized, "usage", None)
            return {
                "content": content,
                "finish_reason": finish,
                "reasoning": _json_safe(getattr(normalized, "reasoning", None)),
                "tool_calls": _tool_calls(normalized),
                "usage": _json_safe(usage),
            }
        if operation == "tool.execute":
            if set(body) != {"id", "name", "arguments"}:
                raise ProcessIntegrationError("tool.execute body is malformed")
            name, args = body["name"], body["arguments"]
            if not isinstance(name, str) or not isinstance(args, Mapping):
                raise ProcessIntegrationError(
                    "tool.execute name/arguments are malformed"
                )
            if name not in self.brokered_tool_names:
                raise ProcessIntegrationError("tool is not host-brokered")
            concrete_registry: ToolRegistry = registry
            with bind_subagent_parent(self.child):
                result = concrete_registry.dispatch_snapshot(
                    self._brokered_dispatch_entries,
                    name,
                    dict(args),
                    task_id=str(getattr(self.child, "_subagent_id", "") or "process"),
                    session_id=getattr(self.child, "session_id", None),
                    user_task=self.task,
                    enabled_tools=set(self.brokered_tool_names),
                )
            return {"result": _json_safe(result)}
        raise ProcessIntegrationError("unsupported broker operation")
