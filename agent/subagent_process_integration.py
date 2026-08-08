"""Host-side adapter for strict profiled subprocess execution."""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import math
import os
import re
import socket
import shutil
import stat
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent.subagent_broker_protocol import SubagentBroker, canonical_json
from agent.subagent_execution_profiles import (
    ExecutionProfileError,
    resolve_execution_profile,
)
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


_MAX_BROKERED_TOOL_CLAIMS = 80
_MAX_BROKERED_TOOL_ARGUMENT_BYTES = 16 * 1024
_MAX_BROKERED_TOOL_RESULT_BYTES = 64 * 1024
_MAX_BROKERED_TOOL_ATTESTATION_BYTES = 8 * 1024
_MAX_BROKERED_TOOL_CLAIM_BYTES = 128 * 1024
_MAX_RESERVED_CLAIM_BYTES = 24 * 1024
_MAX_SCOPED_READ_FILE_BYTES = 16 * 1024 * 1024
_MAX_SCOPED_READ_OUTPUT_CHARS = 100_000
_DEFAULT_CANCELLATION_QUIESCE_SECONDS = 0.5
_CONTAINMENT_REASONS = frozenset({
    "broker-operation-deadline",
    "broker-revocation-deadline",
    "broker-revocation-failed",
})
_STRICT_EVO_SYSTEM_COMMANDS = (
    "git",
    "sh",
    "bash",
    "env",
    "mktemp",
    "awk",
    "grep",
    "mv",
)
_SENSITIVE_CLAIM_KEYS = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "capability_secret",
    "client_secret",
    "credential",
    "credentials",
    "id_token",
    "jwt",
    "key",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
})
_RAW_MODEL_MESSAGE_KEYS = frozenset({
    "conversation_history",
    "messages",
    "model_messages",
    "raw_message",
    "raw_messages",
    "raw_model_messages",
})
_MODEL_MESSAGE_ROLES = frozenset({
    "assistant",
    "developer",
    "system",
    "tool",
    "user",
})
_FORBIDDEN_CLAIM_KEY_FINGERPRINTS = frozenset(
    "".join(character for character in key if character.isalnum())
    for key in (_SENSITIVE_CLAIM_KEYS | _RAW_MODEL_MESSAGE_KEYS)
)


@dataclasses.dataclass(frozen=True)
class BrokeredToolExecutionClaim:
    """Bounded immutable evidence for one host-successful tool execution."""

    sequence: int
    tool_name: str
    arguments_sha256: str
    result_sha256: str
    public_attestation_json: str
    launch_receipt_sha256: str
    tool_schema_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _contains_forbidden_claim_shape(value: Any) -> bool:
    if isinstance(value, Mapping):
        normalized_items = {
            key.lower().replace("-", "_"): item
            for key, item in value.items()
            if isinstance(key, str)
        }
        normalized_keys = set(normalized_items)
        key_fingerprints = {
            "".join(character for character in key if character.isalnum())
            for key in normalized_keys
        }
        if key_fingerprints & _FORBIDDEN_CLAIM_KEY_FINGERPRINTS:
            return True
        if (
            "role" in normalized_keys
            and "content" in normalized_keys
            and str(normalized_items["role"]).lower() in _MODEL_MESSAGE_ROLES
        ):
            return True
        return any(_contains_forbidden_claim_shape(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_claim_shape(item) for item in value)
    return False


def _strict_json_tree(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_strict_json_tree(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _strict_json_tree(item) for key, item in value.items()}
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def _canonical_bounded_json(value: Any, *, field: str, limit: int) -> str:
    try:
        encoded = canonical_json(_strict_json_tree(value))
    except (TypeError, ValueError) as exc:
        raise ProcessIntegrationError(
            f"brokered tool {field} is not canonical JSON"
        ) from exc
    if len(encoded.encode("utf-8")) > limit:
        raise ProcessIntegrationError(f"brokered tool {field} exceeds its byte bound")
    return encoded


def _workspace_relative_parts(raw_path: str, root: Path) -> tuple[str, ...]:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError as exc:
            raise ProcessIntegrationError(
                "Brokered read_file path escapes the profile workspace."
            ) from exc
    parts = tuple(part for part in candidate.parts if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise ProcessIntegrationError(
            "Brokered read_file path escapes or omits the profile workspace target."
        )
    return parts


def _secure_workspace_file_bytes(raw_path: str, workspace_root: str) -> bytes:
    """Read a regular file beneath root using no-follow component opens."""
    if os.open not in os.supports_dir_fd or not getattr(os, "O_NOFOLLOW", 0):
        raise ProcessIntegrationError(
            "Secure brokered read_file is unavailable on this platform."
        )
    root = Path(workspace_root).expanduser().resolve(strict=True)
    parts = _workspace_relative_parts(raw_path, root)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | nofollow_flag
            if index < len(parts) - 1:
                flags |= directory_flag
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProcessIntegrationError(
                "Brokered read_file target is not a regular file."
            )
        if metadata.st_size > _MAX_SCOPED_READ_FILE_BYTES:
            raise ProcessIntegrationError(
                "Brokered read_file target exceeds the secure read byte bound."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(_MAX_SCOPED_READ_FILE_BYTES + 1)
        if len(content) > _MAX_SCOPED_READ_FILE_BYTES:
            raise ProcessIntegrationError(
                "Brokered read_file target exceeds the secure read byte bound."
            )
        return content
    except OSError as exc:
        raise ProcessIntegrationError(
            "Brokered read_file path escapes or is unavailable under the profile workspace."
        ) from exc
    finally:
        os.close(descriptor)


def _workspace_scoped_read_result(
    arguments: Mapping[str, Any], workspace_root: str
) -> str:
    raw_path = arguments.get("path")
    offset = arguments.get("offset", 1)
    limit = arguments.get("limit", 2000)
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ProcessIntegrationError(
            "Brokered read_file requires a non-empty path string."
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        raise ProcessIntegrationError(
            "Brokered read_file offset must be a positive integer."
        )
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > 2000
    ):
        raise ProcessIntegrationError(
            "Brokered read_file limit must be an integer from 1 through 2000."
        )
    content = _secure_workspace_file_bytes(raw_path, workspace_root)
    if b"\x00" in content[:1000]:
        raise ProcessIntegrationError("Brokered read_file target is binary.")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProcessIntegrationError(
            "Brokered read_file target is not UTF-8 text."
        ) from exc
    lines = text.splitlines()
    total_lines = len(lines)
    page = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(
        f"{line_number}|{line}" for line_number, line in enumerate(page, start=offset)
    )
    if len(numbered) > _MAX_SCOPED_READ_OUTPUT_CHARS:
        numbered = numbered[:_MAX_SCOPED_READ_OUTPUT_CHARS]
        truncated_by_chars = True
    else:
        truncated_by_chars = False
    from agent.redact import redact_sensitive_text

    result: dict[str, Any] = {
        "content": redact_sensitive_text(numbered, file_read=True),
        "total_lines": total_lines,
        "file_size": len(content),
        "truncated": truncated_by_chars or total_lines > offset - 1 + len(page),
    }
    if result["truncated"]:
        result["next_offset"] = offset + len(page)
    return json.dumps(result, ensure_ascii=False)


def _require_exact_brokered_workspace(
    arguments: Mapping[str, Any], active_profile: Any
) -> str:
    raw_workspace = arguments.get("workspace")
    if not isinstance(raw_workspace, str) or not raw_workspace:
        raise ProcessIntegrationError(
            "Brokered Scaffolde tool requires an explicit workspace."
        )
    try:
        expected = Path(active_profile.workspace_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProcessIntegrationError(
            "Brokered Scaffolde workspace is unavailable."
        ) from exc
    if (
        getattr(active_profile, "execution_backend", None) == "linux_strict"
        and raw_workspace == "/workspace"
    ):
        return str(expected)
    try:
        supplied_path = Path(raw_workspace).expanduser()
        if not supplied_path.is_absolute():
            supplied_path = expected / supplied_path
        supplied = supplied_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProcessIntegrationError(
            "Brokered Scaffolde workspace is unavailable."
        ) from exc
    if supplied != expected:
        raise ProcessIntegrationError(
            "Brokered Scaffolde workspace does not match the profile workspace."
        )
    return str(expected)


def _expected_scaffolde_attestation_binding(
    arguments: Mapping[str, Any],
    *,
    tool_name: str,
    active_profile: Any,
) -> Mapping[str, Any]:
    if tool_name == "scaffolde_evo_run":
        if active_profile.profile_id != "scaffolde-evo-candidate-v1":
            raise ProcessIntegrationError(
                "brokered Evo run is not bound to the candidate profile"
            )
        expected_request = {
            "experiment_id": arguments.get("experiment_id"),
            "attempt_n": arguments.get("attempt_n"),
        }
        if not all(value is not None for value in expected_request.values()):
            raise ProcessIntegrationError(
                "brokered Evo run request identity is incomplete"
            )
        return MappingProxyType({
            "kind": "evo-run",
            "role": "candidate-worker",
            "profile_id": active_profile.profile_id,
            "protocol_sha256": active_profile.protocol_sha256,
            "request": MappingProxyType(expected_request),
        })
    if tool_name != "scaffolde_evo_agent_dispatch":
        raise ProcessIntegrationError("brokered attestation tool is unsupported")
    role = arguments.get("role")
    profile_ids = {
        "candidate-worker": "scaffolde-evo-candidate-v1",
        "verifier": "scaffolde-evo-verifier-v1",
        "benchmark-reviewer": "scaffolde-evo-benchmark-reviewer-v1",
    }
    profile_id = profile_ids.get(role) if isinstance(role, str) else None
    if profile_id is None:
        raise ProcessIntegrationError("brokered dispatch role is invalid")
    try:
        profile = resolve_execution_profile(profile_id)
    except ExecutionProfileError as exc:
        raise ProcessIntegrationError(
            "brokered dispatch profile could not be resolved"
        ) from exc

    if role == "candidate-worker":
        expected_request = {"parent_node": arguments.get("parent_node")}
    elif role == "verifier":
        expected_request = {
            "experiment_id": arguments.get("experiment_id"),
            "phase": arguments.get("phase"),
            "attempt_n": arguments.get("attempt_n"),
        }
    else:
        expected_request = {
            "experiment_id": arguments.get("experiment_id"),
            "mode": arguments.get("mode", "review-experiment"),
            "attempt_n": arguments.get("attempt_n"),
        }
    if not all(value is not None for value in expected_request.values()):
        raise ProcessIntegrationError(
            "brokered dispatch request identity is incomplete"
        )
    return MappingProxyType({
        "kind": None,
        "role": role,
        "profile_id": profile_id,
        "protocol_sha256": profile.protocol_sha256,
        "request": MappingProxyType(expected_request),
    })


def _canonical_public_attestation_json(
    value: Any,
    *,
    expected_binding: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> str:
    if not isinstance(value, Mapping):
        raise ProcessIntegrationError(
            "brokered tool result must contain exactly one top-level "
            "broker_attestation mapping"
        )
    expected_kind = expected_binding["kind"]
    expected_keys = {
        "version",
        "role",
        "profile_id",
        "protocol_sha256",
        "ok",
        "request",
        "completion",
        "outcome",
        "evidence_sha256",
    }
    if expected_kind is not None:
        expected_keys.add("kind")
    if expected_kind == "evo-run":
        expected_keys.add("attempt_execution")
    if set(value) != expected_keys:
        raise ProcessIntegrationError(
            "brokered tool public attestation does not match the Scaffolde schema"
        )
    if isinstance(value.get("version"), bool) or value.get("version") != 1:
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid version"
        )
    if expected_kind is not None and value.get("kind") != expected_kind:
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid kind"
        )
    host_result: Mapping[str, Any] | None = None
    if expected_kind == "evo-run":
        attempt_execution = value.get("attempt_execution")
        if not isinstance(attempt_execution, Mapping) or set(attempt_execution) != {
            "backend",
            "containment_mode",
            "authority_sha256",
            "executable_sha256",
            "cleanup_sha256",
        }:
            raise ProcessIntegrationError(
                "brokered tool public attestation has an invalid attempt execution"
            )
        if (
            attempt_execution.get("backend") != "linux-strict"
            or attempt_execution.get("containment_mode")
            != "linux-strict-bwrap-cgroup-v2"
        ):
            raise ProcessIntegrationError(
                "brokered tool public attestation has an invalid attempt containment"
            )
        for field in ("authority_sha256", "executable_sha256", "cleanup_sha256"):
            digest = attempt_execution.get(field)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ProcessIntegrationError(
                    f"brokered tool public attestation has an invalid {field}"
                )
        expected_payload_keys = {
            "ok",
            "broker_attestation",
            "completion",
            "result",
            "execution_receipt",
            "execution_receipt_hash",
            "evidence",
        }
        if set(result_payload) != expected_payload_keys:
            raise ProcessIntegrationError(
                "brokered Evo run result does not match the exact producer schema"
            )
        host_receipt = result_payload.get("execution_receipt")
        expected_receipt_keys = {
            "version",
            "runner",
            "backend",
            "containment_mode",
            "state",
            "authority_sha256",
            "executable_sha256",
            "argv_sha256",
            "root_pid",
            "exit_code",
            "timed_out",
            "cancelled",
            "cleanup",
        }
        if (
            not isinstance(host_receipt, Mapping)
            or set(host_receipt) != expected_receipt_keys
        ):
            raise ProcessIntegrationError(
                "brokered Evo run has an invalid host execution receipt"
            )
        host_cleanup = host_receipt.get("cleanup")
        expected_cleanup_keys = {
            "root_reaped",
            "process_group_empty",
            "cgroup_kill_sent",
            "cgroup_empty",
            "cgroup_removed",
            "broker_quiesced",
        }
        if (
            not isinstance(host_cleanup, Mapping)
            or set(host_cleanup) != expected_cleanup_keys
            or host_cleanup.get("root_reaped") is not True
            or host_cleanup.get("process_group_empty") is not True
            or not isinstance(host_cleanup.get("cgroup_kill_sent"), bool)
            or host_cleanup.get("cgroup_empty") is not True
            or host_cleanup.get("cgroup_removed") is not True
            or host_cleanup.get("broker_quiesced") is not None
        ):
            raise ProcessIntegrationError(
                "brokered Evo run host cleanup is not complete"
            )
        if (
            host_receipt.get("version") != 2
            or host_receipt.get("runner") != "scaffolde-evo-run"
            or host_receipt.get("backend") != attempt_execution.get("backend")
            or host_receipt.get("containment_mode")
            != attempt_execution.get("containment_mode")
            or host_receipt.get("state") != "SUCCEEDED"
            or host_receipt.get("authority_sha256")
            != attempt_execution.get("authority_sha256")
            or host_receipt.get("executable_sha256")
            != attempt_execution.get("executable_sha256")
            or not isinstance(host_receipt.get("argv_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", host_receipt["argv_sha256"])
            or isinstance(host_receipt.get("root_pid"), bool)
            or not isinstance(host_receipt.get("root_pid"), int)
            or host_receipt["root_pid"] < 1
            or isinstance(host_receipt.get("exit_code"), bool)
            or host_receipt.get("exit_code") != 0
            or host_receipt.get("timed_out") is not False
            or host_receipt.get("cancelled") is not False
        ):
            raise ProcessIntegrationError(
                "brokered tool public attestation does not match host execution receipt"
            )
        cleanup_sha256 = hashlib.sha256(
            canonical_json(_json_safe(host_cleanup)).encode("utf-8")
        ).hexdigest()
        if attempt_execution.get("cleanup_sha256") != cleanup_sha256:
            raise ProcessIntegrationError(
                "brokered tool public attestation does not match host cleanup"
            )
        host_receipt_sha256 = hashlib.sha256(
            canonical_json(_json_safe(host_receipt)).encode("utf-8")
        ).hexdigest()
        completion = value.get("completion")
        if (
            result_payload.get("execution_receipt_hash") != host_receipt_sha256
            or not isinstance(completion, Mapping)
            or completion.get("execution_receipt_hash") != host_receipt_sha256
        ):
            raise ProcessIntegrationError(
                "brokered Evo run execution receipt hash does not match its receipt"
            )
        host_result = result_payload.get("result")
        host_completion = result_payload.get("completion")
        if not isinstance(host_result, Mapping) or not isinstance(
            host_completion, Mapping
        ):
            raise ProcessIntegrationError("brokered Evo run result is malformed")
        evidence = result_payload.get("evidence")
        digest_fields = {
            "argv_sha256",
            "before_node_sha256",
            "after_node_sha256",
            "stdout_sha256",
            "stderr_sha256",
        }
        if not isinstance(evidence, Mapping) or set(evidence) != digest_fields | {
            "stdout_truncated",
            "stderr_truncated",
        }:
            raise ProcessIntegrationError("brokered Evo run evidence is malformed")
        if any(
            not isinstance(evidence.get(field), str)
            or not re.fullmatch(r"[0-9a-f]{64}", evidence[field])
            for field in digest_fields
        ):
            raise ProcessIntegrationError("brokered Evo run evidence digest is invalid")
        if not all(
            isinstance(evidence.get(field), bool)
            for field in ("stdout_truncated", "stderr_truncated")
        ):
            raise ProcessIntegrationError(
                "brokered Evo run truncation evidence is invalid"
            )
        evidence_sha256 = hashlib.sha256(
            canonical_json(_json_safe(evidence)).encode("utf-8")
        ).hexdigest()
        if value.get("evidence_sha256") != evidence_sha256:
            raise ProcessIntegrationError(
                "brokered Evo run evidence hash does not match its evidence"
            )
        host_result_sha256 = hashlib.sha256(
            canonical_json(_json_safe(host_result)).encode("utf-8")
        ).hexdigest()
        if (
            host_completion.get("result_hash") != host_result_sha256
            or completion.get("result_hash") != host_result_sha256
        ):
            raise ProcessIntegrationError(
                "brokered Evo run result hash does not match its result"
            )
    if value.get("ok") is not True:
        raise ProcessIntegrationError(
            "brokered tool public attestation does not attest success"
        )
    role = value.get("role")
    profile_id = value.get("profile_id")
    request = value.get("request")
    if expected_kind == "evo-run":
        role_contracts = {
            "candidate-worker": (
                "scaffolde-evo-candidate-v1",
                {"experiment_id", "attempt_n"},
            ),
        }
    else:
        role_contracts = {
            "candidate-worker": (
                "scaffolde-evo-candidate-v1",
                {"parent_node"},
            ),
            "verifier": (
                "scaffolde-evo-verifier-v1",
                {"experiment_id", "phase", "attempt_n"},
            ),
            "benchmark-reviewer": (
                "scaffolde-evo-benchmark-reviewer-v1",
                {
                    "experiment_id",
                    "mode",
                    "attempt_n",
                },
            ),
        }
    role_contract = role_contracts.get(role) if isinstance(role, str) else None
    if (
        role_contract is None
        or profile_id != role_contract[0]
        or not isinstance(request, Mapping)
    ):
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid role, profile, or request"
        )
    if set(request) != role_contract[1]:
        raise ProcessIntegrationError(
            "brokered tool public attestation request does not match its profile"
        )
    if (
        role != expected_binding["role"]
        or profile_id != expected_binding["profile_id"]
        or value.get("protocol_sha256") != expected_binding["protocol_sha256"]
    ):
        raise ProcessIntegrationError(
            "brokered tool public attestation does not match the host binding"
        )
    expected_request = expected_binding["request"]
    if any(request.get(key) != item for key, item in expected_request.items()):
        raise ProcessIntegrationError(
            "brokered tool public attestation does not match the dispatched request"
        )
    node_field = "parent_node" if "parent_node" in request else "experiment_id"
    node_id = request.get(node_field)
    if not isinstance(node_id, str) or not re.fullmatch(r"exp_\d{4,}", node_id):
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid experiment identity"
        )
    if role == "verifier":
        if request.get("phase") not in {"pre", "post"}:
            raise ProcessIntegrationError(
                "brokered tool public attestation has an invalid verifier phase"
            )
    elif role == "benchmark-reviewer":
        if request.get("mode") != "review-experiment":
            raise ProcessIntegrationError(
                "brokered tool public attestation has an invalid reviewer mode"
            )
    if "attempt_n" in request and (
        isinstance(request.get("attempt_n"), bool)
        or not isinstance(request.get("attempt_n"), int)
        or request["attempt_n"] < 1
    ):
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid attempt number"
        )
    outcome = value.get("outcome")
    if expected_kind == "evo-run":
        outcome_valid = (
            isinstance(outcome, Mapping)
            and set(outcome) == {"status", "exit_code"}
            and outcome.get("status") in {"committed", "evaluated", "failed"}
            and not isinstance(outcome.get("exit_code"), bool)
            and isinstance(outcome.get("exit_code"), int)
            and (outcome.get("status") == "failed") is (outcome.get("exit_code") != 0)
        )
    elif role == "candidate-worker":
        outcome_valid = (
            isinstance(outcome, Mapping)
            and set(outcome) == {"validated"}
            and outcome.get("validated") is True
        )
    elif role == "verifier":
        outcome_valid = (
            isinstance(outcome, Mapping)
            and set(outcome) == {"passed", "verdict"}
            and isinstance(outcome.get("passed"), bool)
            and outcome.get("verdict") in {"pass", "warn", "fail"}
            and outcome.get("passed") is (outcome.get("verdict") != "fail")
        )
    else:
        outcome_valid = (
            isinstance(outcome, Mapping)
            and set(outcome) == {"reviewed"}
            and outcome.get("reviewed") is True
        )
    if not outcome_valid:
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid role outcome"
        )
    if expected_kind == "evo-run":
        if (
            not isinstance(host_result, Mapping)
            or not isinstance(outcome, Mapping)
            or set(host_result) != {"experiment_id", "attempt_n", "status", "exit_code"}
            or host_result.get("experiment_id") != request.get("experiment_id")
            or host_result.get("attempt_n") != request.get("attempt_n")
            or host_result.get("status") != outcome.get("status")
            or host_result.get("exit_code") != outcome.get("exit_code")
        ):
            raise ProcessIntegrationError(
                "brokered Evo run result does not match its request and outcome"
            )
    for field in ("protocol_sha256", "evidence_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProcessIntegrationError(
                f"brokered tool public attestation has an invalid {field}"
            )
    completion = value.get("completion")
    if not isinstance(completion, Mapping) or set(completion) != {
        "result_hash",
        "execution_receipt_hash",
    }:
        raise ProcessIntegrationError(
            "brokered tool public attestation has an invalid completion binding"
        )
    for field in ("result_hash", "execution_receipt_hash"):
        digest = completion.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProcessIntegrationError(
                f"brokered tool public attestation has an invalid completion {field}"
            )
    host_completion = result_payload.get("completion")
    host_result_hash = (
        host_completion.get("result_hash")
        if isinstance(host_completion, Mapping)
        else None
    )
    host_execution_receipt_hash = result_payload.get("execution_receipt_hash")
    host_evidence = result_payload.get("evidence")
    if (
        not isinstance(host_completion, Mapping)
        or not isinstance(host_evidence, Mapping)
        or not isinstance(host_result_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", host_result_hash)
        or not isinstance(host_execution_receipt_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", host_execution_receipt_hash)
        or completion.get("result_hash") != host_result_hash
        or completion.get("execution_receipt_hash") != host_execution_receipt_hash
        or value.get("evidence_sha256")
        != hashlib.sha256(
            canonical_json(_json_safe(host_evidence)).encode("utf-8")
        ).hexdigest()
    ):
        raise ProcessIntegrationError(
            "brokered tool public attestation does not match host completion evidence"
        )
    encoded = _canonical_bounded_json(
        value,
        field="public attestation",
        limit=_MAX_BROKERED_TOOL_ATTESTATION_BYTES,
    )
    if _contains_forbidden_claim_shape(value):
        raise ProcessIntegrationError(
            "brokered tool public attestation contains forbidden sensitive data"
        )
    from agent.redact import redact_sensitive_text

    if (
        redact_sensitive_text(encoded, force=True, redact_url_credentials=True)
        != encoded
    ):
        raise ProcessIntegrationError(
            "brokered tool public attestation contains forbidden sensitive data"
        )
    return encoded


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProcessIntegrationError(
                "brokered tool result contains a duplicate JSON key"
            )
        value[key] = item
    return value


def _reject_non_finite_json_constant(_value: str) -> None:
    raise ProcessIntegrationError("brokered tool result is not canonical JSON")


def _brokered_result_payload(result: Any) -> tuple[Mapping[str, Any], Any]:
    if isinstance(result, str):
        try:
            payload = json.loads(
                result,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_finite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise ProcessIntegrationError(
                "brokered tool result is malformed JSON"
            ) from exc
        response_result = result
    elif isinstance(result, Mapping):
        payload = result
        response_result = result
    else:
        raise ProcessIntegrationError(
            "brokered tool result must be a JSON string or mapping"
        )
    if not isinstance(payload, Mapping):
        raise ProcessIntegrationError(
            "brokered tool result must contain exactly one top-level "
            "broker_attestation mapping"
        )
    if payload.get("ok") is False:
        if "broker_attestation" in payload:
            raise ProcessIntegrationError(
                "failed brokered tool result must not contain a broker_attestation"
            )
        return payload, response_result
    attestation = payload.get("broker_attestation")
    if "broker_attestation" not in payload or not isinstance(attestation, Mapping):
        raise ProcessIntegrationError(
            "brokered tool result must contain exactly one top-level "
            "broker_attestation mapping"
        )
    return payload, response_result


def _bounded_diagnostic_text(value: object, *, limit: int) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(value)
    )
    return " ".join(printable.split())[:limit]


def _bounded_failed_tool_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    from agent.redact import redact_sensitive_text

    error_code = _bounded_diagnostic_text(
        payload.get("error_code") or "brokered_tool_failed", limit=80
    )
    error = _bounded_diagnostic_text(
        payload.get("error") or "Brokered tool failed", limit=500
    )
    return {
        "ok": False,
        "error_code": error_code,
        "error": redact_sensitive_text(error, force=True, redact_url_credentials=True),
    }


def strict_worker_runtime_mounts(
    *, expose_scaffolde_evo_run: bool = False
) -> tuple[Any, ...]:
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
    if expose_scaffolde_evo_run:
        for command in _STRICT_EVO_SYSTEM_COMMANDS:
            executable = shutil.which(command)
            if executable is None:
                raise ProcessIntegrationError(
                    f"strict Evo runtime requires host utility {command}"
                )
            command_path = Path(executable).absolute()
            path = command_path.resolve(strict=True)
            mounts.setdefault(str(path), RuntimeMount(source=path, target=path))
            mounts.setdefault(
                str(command_path),
                RuntimeMount(source=path, target=command_path),
            )
        evo_executable = shutil.which("evo")
        if evo_executable is None:
            raise ProcessIntegrationError("strict Evo runtime requires the Evo CLI")
        evo_command = Path(evo_executable).absolute()
        evo_path = evo_command.resolve(strict=True)
        # uv/pip tool installs keep their interpreter and packages beside the
        # entry point. Mount that declared tool environment, never its home.
        evo_runtime = evo_path.parent.parent
        mounts.setdefault(
            str(evo_runtime),
            RuntimeMount(source=evo_runtime, target=evo_runtime),
        )
        if not _runtime_target_is_covered(mounts, evo_command):
            mounts.setdefault(
                str(evo_command),
                RuntimeMount(source=evo_path, target=evo_command),
            )
        git_core = Path("/usr/lib/git-core")
        if git_core.is_dir():
            mounts.setdefault(
                str(git_core),
                RuntimeMount(source=git_core.resolve(strict=True), target=git_core),
            )
    return tuple(mounts.values())


def _runtime_target_is_covered(mounts: Mapping[str, Any], target: Path) -> bool:
    """Return whether an identical-path directory mount already exposes target."""

    return any(
        mount.source == mount.target
        and mount.source.is_dir()
        and target.is_relative_to(mount.target)
        for mount in mounts.values()
    )


def strict_worker_runtime_path(*, expose_scaffolde_evo_run: bool = False) -> str:
    """Return a narrow PATH matching host-declared strict runtime mounts."""
    directories = {str(Path(sys.executable).resolve(strict=True).parent)}
    if expose_scaffolde_evo_run:
        for command in ("evo", *_STRICT_EVO_SYSTEM_COMMANDS):
            executable = shutil.which(command)
            if executable is None:
                raise ProcessIntegrationError(
                    f"strict Evo runtime requires host utility {command}"
                )
            directories.add(str(Path(executable).absolute().parent))
    return os.pathsep.join(sorted(directories))


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
        self,
        *,
        broker: SubagentBroker,
        child: Any,
        profile: Any,
        task: str,
        expected_tool_schema_sha256: str | None = None,
    ) -> None:
        self.broker = broker
        self.child = child
        self.profile = profile
        self.task = task
        self._cancellation_event = threading.Event()
        self._secret = broker.reveal_secret_for_transport()
        self._launch_receipt_sha256 = broker.launch_receipt_digest
        pinned_tools_json = getattr(
            child, "_delegate_provider_effective_tools_json", None
        )
        if pinned_tools_json is None:
            self._effective_tools = self._current_provider_effective_tools()
        else:
            try:
                self._effective_tools = json.loads(pinned_tools_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ProcessIntegrationError(
                    "pinned provider-effective tool schema is malformed"
                ) from exc
        self._tool_schema_sha256 = exact_tool_schema_digest(self._effective_tools)
        if (
            expected_tool_schema_sha256 is not None
            and self._tool_schema_sha256 != expected_tool_schema_sha256
        ):
            raise ProcessIntegrationError(
                "provider-effective tool schema does not match the launch receipt"
            )
        if (
            exact_tool_schema_digest(self._current_provider_effective_tools())
            != self._tool_schema_sha256
        ):
            raise ProcessIntegrationError(
                "provider-effective tool schema changed after launch receipt"
            )
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
        self._claim_lock = threading.Lock()
        self._tool_execution_claims: tuple[BrokeredToolExecutionClaim, ...] = ()
        self._tool_execution_claim_bytes = 2  # Canonical JSON list brackets.
        self._pending_tool_execution_claims = 0
        self._claims_frozen = False
        self._claim_cleanup_failure: str | None = None
        self._side_effects_unresolved = False
        self._unresolved_operation_label: str | None = None
        self._stop_requested: threading.Event | None = None
        self._operation_lock = threading.Lock()
        self._operation_state_lock = threading.Lock()
        self._active_operation_label: str | None = None
        if set(self._brokered_dispatch_entries) != set(self.brokered_tool_names):
            raise ProcessIntegrationError("host-brokered frozen handler is unavailable")

    def _current_provider_effective_tools(self) -> list[Mapping[str, Any]]:
        kwargs = self.child._build_api_kwargs(
            [], tools_for_api=copy.deepcopy(self.child.tools)
        )
        tools = _json_safe(kwargs.get("tools"))
        if not isinstance(tools, list) or not all(
            isinstance(tool, Mapping) for tool in tools
        ):
            raise ProcessIntegrationError(
                "provider-effective tool schema is unavailable or malformed"
            )
        return tools

    @property
    def tool_execution_claims(self) -> tuple[BrokeredToolExecutionClaim, ...]:
        with self._claim_lock:
            return self._tool_execution_claims

    @property
    def claims_frozen(self) -> bool:
        with self._claim_lock:
            return self._claims_frozen

    @property
    def claim_cleanup_failure(self) -> str | None:
        with self._claim_lock:
            return self._claim_cleanup_failure

    @property
    def side_effects_unresolved(self) -> bool:
        with self._claim_lock:
            return self._side_effects_unresolved

    @property
    def unresolved_operation_label(self) -> str | None:
        with self._claim_lock:
            return self._unresolved_operation_label

    @property
    def active_operation_label(self) -> str | None:
        with self._operation_state_lock:
            return self._active_operation_label

    def _dispatch_admitted(
        self, operation: str, body: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        label = operation
        if operation == "tool.execute":
            name = body.get("name")
            if isinstance(name, str) and name in self.brokered_tool_names:
                label = f"{operation}:{name}"
        with self._operation_lock:
            with self._operation_state_lock:
                self._active_operation_label = label
            try:
                return self._dispatch(operation, body)
            finally:
                with self._operation_state_lock:
                    self._active_operation_label = None

    def _freeze_claim_accounting(self, failure: str | None = None) -> None:
        with self._claim_lock:
            self._claims_frozen = True
            self._pending_tool_execution_claims = 0
            if failure is not None and self._claim_cleanup_failure is None:
                self._claim_cleanup_failure = failure

    def _reserve_claim_capacity(self) -> None:
        with self._claim_lock:
            if self._claims_frozen:
                raise ProcessIntegrationError(
                    "brokered tool claim accounting is frozen"
                )
            if (
                len(self._tool_execution_claims) + self._pending_tool_execution_claims
                >= _MAX_BROKERED_TOOL_CLAIMS
            ):
                raise ProcessIntegrationError(
                    "brokered tool claim count exceeds its bound"
                )
            reserved_bytes = (self._pending_tool_execution_claims + 1) * (
                _MAX_RESERVED_CLAIM_BYTES + 1
            )
            if (
                self._tool_execution_claim_bytes + reserved_bytes
                > _MAX_BROKERED_TOOL_CLAIM_BYTES
            ):
                raise ProcessIntegrationError(
                    "brokered tool claims exceed their aggregate byte bound"
                )
            self._pending_tool_execution_claims += 1

    def _release_claim_capacity(self) -> None:
        with self._claim_lock:
            if self._claims_frozen:
                return
            if self._pending_tool_execution_claims <= 0:
                raise ProcessIntegrationError(
                    "brokered tool claim reservation is unavailable"
                )
            self._pending_tool_execution_claims -= 1

    def _record_tool_execution_claim(
        self,
        *,
        name: str,
        arguments_sha256: str,
        result_sha256: str,
        public_attestation_json: str,
    ) -> None:
        with self._claim_lock:
            if self._claims_frozen:
                raise ProcessIntegrationError(
                    "brokered tool claim accounting is frozen"
                )
            if self._pending_tool_execution_claims <= 0:
                raise ProcessIntegrationError(
                    "brokered tool claim reservation is unavailable"
                )
            if len(self._tool_execution_claims) >= _MAX_BROKERED_TOOL_CLAIMS:
                raise ProcessIntegrationError(
                    "brokered tool claim count exceeds its bound"
                )
            claim = BrokeredToolExecutionClaim(
                sequence=len(self._tool_execution_claims) + 1,
                tool_name=name,
                arguments_sha256=arguments_sha256,
                result_sha256=result_sha256,
                public_attestation_json=public_attestation_json,
                launch_receipt_sha256=self._launch_receipt_sha256,
                tool_schema_sha256=self._tool_schema_sha256,
            )
            claim_bytes = len(canonical_json(claim.to_dict()).encode("utf-8"))
            if claim_bytes > _MAX_RESERVED_CLAIM_BYTES:
                raise ProcessIntegrationError(
                    "brokered tool claim exceeds its reserved byte bound"
                )
            separator_bytes = 1 if self._tool_execution_claims else 0
            remaining_reserved_bytes = (self._pending_tool_execution_claims - 1) * (
                _MAX_RESERVED_CLAIM_BYTES + 1
            )
            if (
                self._tool_execution_claim_bytes
                + separator_bytes
                + claim_bytes
                + remaining_reserved_bytes
                > _MAX_BROKERED_TOOL_CLAIM_BYTES
            ):
                raise ProcessIntegrationError(
                    "brokered tool claims exceed their aggregate byte bound"
                )
            self._tool_execution_claims = (*self._tool_execution_claims, claim)
            self._tool_execution_claim_bytes += separator_bytes + claim_bytes
            self._pending_tool_execution_claims -= 1

    def serve(
        self,
        channel: socket.socket,
        *,
        root_pid: int,
        stop_requested: threading.Event,
    ) -> None:
        del root_pid
        self._stop_requested = stop_requested
        self.child._owned_process_broker_stop_requested = stop_requested
        self.child._owned_process_broker_cancellation_event = self._cancellation_event
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
                body = self._dispatch_admitted(request.operation, request.body)
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
                elif isinstance(exc, ProcessIntegrationError):
                    from agent.redact import redact_sensitive_text

                    detail = redact_sensitive_text(
                        _bounded_diagnostic_text(exc, limit=500),
                        force=True,
                        redact_url_credentials=True,
                    )
                    if detail:
                        error = f"{error}: {detail}"
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

    def cancel(
        self, *, timeout_seconds: float = _DEFAULT_CANCELLATION_QUIESCE_SECONDS
    ) -> bool:
        """Revoke admission and wait only briefly for accepted host work."""
        if (
            not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and non-negative")
        if self._stop_requested is not None:
            self._stop_requested.set()
        self._cancellation_event.set()
        self.broker.revoke("owned process broker cancellation requested")
        # The process-profile child owns its provider client. Closing that
        # transport is the cancellation path for a blocking model.complete;
        # lifecycle-aware tool handlers are cancelled by the broker stop event
        # observed in SubagentLifecycleService.wait().
        try:
            close_client = getattr(getattr(self.child, "client", None), "close", None)
            if callable(close_client):
                close_client()
        except Exception:
            pass
        try:
            from agent.interrupt_compat import request_hard_interrupt

            request_hard_interrupt(self.child, "Owned process broker cancelled")
        except Exception:
            pass
        # An admitted host operation can be blocked in provider/plugin code that
        # Python cannot forcibly stop. Never let revocation wait on that lock
        # forever: False tells the owned runner to terminalize fail-closed.
        operation_quiesced = self._operation_lock.acquire(timeout=timeout_seconds)
        if operation_quiesced:
            self._operation_lock.release()
            self._freeze_claim_accounting()
        return operation_quiesced

    def finalize(self) -> None:
        """Freeze claims only after the owned runner has proved quiescence."""
        with self._operation_lock:
            self._freeze_claim_accounting()

    def _dispatch(self, operation: str, body: Any) -> Mapping[str, Any]:
        if not isinstance(body, Mapping):
            raise ProcessIntegrationError("broker operation body must be an object")
        if operation == "session.start":
            if body:
                raise ProcessIntegrationError("session.start body must be empty")
            if (
                exact_tool_schema_digest(self._current_provider_effective_tools())
                != self._tool_schema_sha256
            ):
                raise ProcessIntegrationError(
                    "provider-effective tool schema changed after launch"
                )
            return {
                "protocol": self.profile.protocol_text,
                "task": self.task,
                "tools": _json_safe(self._effective_tools),
                "tool_schema_digest": self._tool_schema_sha256,
                "local_tool_names": sorted(self.local_tool_names),
                "brokered_tool_names": sorted(self.brokered_tool_names),
                "model": str(self.child.model or ""),
                "max_iterations": self.profile.max_process_iterations,
            }
        if operation == "model.complete":
            if set(body) != {"messages"} or not isinstance(body["messages"], list):
                raise ProcessIntegrationError("model.complete body is malformed")
            kwargs = self.child._build_api_kwargs(
                body["messages"], tools_for_api=copy.deepcopy(self.child.tools)
            )
            effective_tools = _json_safe(kwargs.get("tools"))
            if (
                not isinstance(effective_tools, list)
                or not all(isinstance(tool, Mapping) for tool in effective_tools)
                or exact_tool_schema_digest(effective_tools) != self._tool_schema_sha256
            ):
                raise ProcessIntegrationError(
                    "provider-effective tool schema changed after launch"
                )
            kwargs["tools"] = copy.deepcopy(self._effective_tools)
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
            if (
                exact_tool_schema_digest(self._current_provider_effective_tools())
                != self._tool_schema_sha256
            ):
                raise ProcessIntegrationError(
                    "effective tool schema changed after launch"
                )
            if name == "read_file":
                try:
                    result = _workspace_scoped_read_result(
                        args, self.profile.workspace_root
                    )
                except ProcessIntegrationError:
                    return {
                        "result": {
                            "ok": False,
                            "error_code": "workspace_path_invalid",
                            "error": "read_file path must be available under the profile workspace",
                        }
                    }
                return {"result": result}
            if name in {
                "scaffolde_evo_agent_dispatch",
                "scaffolde_evo_run",
            }:
                host_workspace = _require_exact_brokered_workspace(args, self.profile)
                dispatch_args = {**args, "workspace": host_workspace}
            else:
                dispatch_args = dict(args)
            arguments_json = _canonical_bounded_json(
                dispatch_args,
                field="arguments",
                limit=_MAX_BROKERED_TOOL_ARGUMENT_BYTES,
            )
            arguments_sha256 = hashlib.sha256(
                arguments_json.encode("utf-8")
            ).hexdigest()
            self._reserve_claim_capacity()
            claim_recorded = False
            side_effecting = name in {
                "scaffolde_evo_agent_dispatch",
                "scaffolde_evo_run",
            }
            prior_side_effects_unresolved = False
            prior_unresolved_operation_label: str | None = None
            if side_effecting:
                with self._claim_lock:
                    prior_side_effects_unresolved = self._side_effects_unresolved
                    prior_unresolved_operation_label = self._unresolved_operation_label
                    self._side_effects_unresolved = True
                    self._unresolved_operation_label = f"tool.execute:{name}"
            try:
                concrete_registry: ToolRegistry = registry
                with bind_subagent_parent(self.child):
                    cancellation_kwargs = (
                        {"cancellation_event": self._cancellation_event}
                        if name == "scaffolde_evo_run"
                        else {}
                    )
                    result = concrete_registry.dispatch_snapshot(
                        self._brokered_dispatch_entries,
                        name,
                        dispatch_args,
                        task_id=str(
                            getattr(self.child, "_subagent_id", "") or "process"
                        ),
                        session_id=getattr(self.child, "session_id", None),
                        user_task=self.task,
                        enabled_tools=set(self.brokered_tool_names),
                        **cancellation_kwargs,
                    )
                result_payload, response_result = _brokered_result_payload(result)
                result_json = _canonical_bounded_json(
                    result_payload,
                    field="result",
                    limit=_MAX_BROKERED_TOOL_RESULT_BYTES,
                )
                if result_payload.get("ok") is False:
                    with self._claim_lock:
                        if result_payload.get("side_effects_unresolved") is True:
                            self._side_effects_unresolved = True
                            completion = result_payload.get("completion")
                            nested_operation = (
                                completion.get("active_operation")
                                if isinstance(completion, Mapping)
                                else None
                            )
                            if nested_operation is None and isinstance(
                                completion, Mapping
                            ):
                                lifecycle_stage = completion.get("lifecycle_stage")
                                if isinstance(lifecycle_stage, str):
                                    nested_operation = f"stage:{lifecycle_stage}"
                            if nested_operation is None and isinstance(
                                completion, Mapping
                            ):
                                containment_reason = completion.get(
                                    "containment_reason"
                                )
                                if containment_reason in _CONTAINMENT_REASONS:
                                    nested_operation = (
                                        f"containment:{containment_reason}"
                                    )
                            if (
                                isinstance(nested_operation, str)
                                and 1 <= len(nested_operation) <= 128
                                and all(
                                    character.isascii()
                                    and (character.isalnum() or character in "._:-")
                                    for character in nested_operation
                                )
                            ):
                                label = f"tool.execute:{name}:{nested_operation}"
                                if len(label) <= 128:
                                    self._unresolved_operation_label = label
                        elif (
                            side_effecting
                            and result_payload.get("side_effects_unresolved") is False
                        ):
                            self._side_effects_unresolved = (
                                prior_side_effects_unresolved
                            )
                            self._unresolved_operation_label = (
                                prior_unresolved_operation_label
                            )
                    return {"result": _bounded_failed_tool_result(result_payload)}
                expected_attestation_binding = _expected_scaffolde_attestation_binding(
                    args,
                    tool_name=name,
                    active_profile=self.profile,
                )
                public_attestation_json = _canonical_public_attestation_json(
                    result_payload["broker_attestation"],
                    expected_binding=expected_attestation_binding,
                    result_payload=result_payload,
                )
                self._record_tool_execution_claim(
                    name=name,
                    arguments_sha256=arguments_sha256,
                    result_sha256=hashlib.sha256(
                        result_json.encode("utf-8")
                    ).hexdigest(),
                    public_attestation_json=public_attestation_json,
                )
                claim_recorded = True
                if side_effecting:
                    with self._claim_lock:
                        self._side_effects_unresolved = prior_side_effects_unresolved
                        self._unresolved_operation_label = (
                            prior_unresolved_operation_label
                        )
                return {"result": _json_safe(response_result)}
            finally:
                if not claim_recorded:
                    self._release_claim_capacity()
        raise ProcessIntegrationError("unsupported broker operation")
