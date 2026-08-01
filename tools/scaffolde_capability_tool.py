"""Native narrow-waist tool for Scaffolde capability descriptors."""

from __future__ import annotations

import json
from typing import Any, Dict

from tools.registry import registry
from tools.scaffolde_capabilities import invoke_capability, load_capability_registry


SCAFFOLDE_CAPABILITY_SCHEMA = {
    "name": "scaffolde_capability",
    "description": (
        "Authoritative native runtime for Scaffolde-owned capabilities and PAI accounts. "
        "Use action=status to inspect descriptor health, action=list to see available capabilities/operations, "
        "and action=invoke with capability_id, operation, and arguments for execution. Prefer this over generic "
        "Gmail/Google clients for Scaffolde-owned or pai@scaffolde.ai requests unless the user explicitly overrides."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "list", "invoke"], "description": "Operation on the Scaffolde capability registry."},
            "capability_id": {"type": "string", "description": "Capability id, e.g. scaffolde.gmail.pai. Required for invoke."},
            "operation": {"type": "string", "description": "Descriptor operation to invoke. Required for invoke."},
            "arguments": {"type": "object", "description": "Exact descriptor-declared arguments for the operation; no extra keys."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _status_name(raw_status: str) -> str:
    if raw_status == "valid":
        return "ok"
    if raw_status == "absent":
        return "absent"
    return "capability_degraded"


def handle_scaffolde_capability(args: Dict[str, Any], **_kwargs) -> str:
    action = (args or {}).get("action")
    reg = load_capability_registry()
    if action == "status":
        payload = reg.as_dict()
        payload["status"] = _status_name(reg.status)
        payload["raw_status"] = reg.status
        return json.dumps(payload, ensure_ascii=False)
    if action == "list":
        payload = {
            "status": _status_name(reg.status),
            "raw_status": reg.status,
            "capabilities": [],
            "errors": reg.errors,
            "path": reg.path,
        }
        for cap in reg.capabilities.values():
            payload["capabilities"].append({
                "id": cap.get("id"),
                "tool_name": cap.get("tool_name"),
                "authority": cap.get("authority"),
                "kind": cap.get("kind"),
                "description": cap.get("description"),
                "triggers": cap.get("triggers", []),
                "operations": {
                    name: {
                        "risk": op.get("risk"),
                        "parameters": op.get("parameters", {}),
                    }
                    for name, op in cap.get("operations", {}).items()
                },
            })
        return json.dumps(payload, ensure_ascii=False)
    if action == "invoke":
        result = invoke_capability(
            str((args or {}).get("capability_id") or ""),
            str((args or {}).get("operation") or ""),
            (args or {}).get("arguments") or {},
        )
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"status": "error", "error_type": "invalid_action", "message": "action must be status, list, or invoke"}, ensure_ascii=False)


registry.register(
    name="scaffolde_capability",
    toolset="scaffolde_capability",
    schema=SCAFFOLDE_CAPABILITY_SCHEMA,
    handler=handle_scaffolde_capability,
    description=SCAFFOLDE_CAPABILITY_SCHEMA["description"],
    emoji="🏗️",
)
