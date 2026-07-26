"""Prompt guidance for Scaffolde native capabilities."""

from __future__ import annotations

from typing import Iterable

from tools.scaffolde_capabilities import load_capability_registry


def build_scaffolde_capabilities_prompt(valid_tool_names: Iterable[str]) -> str:
    if "scaffolde_capability" not in set(valid_tool_names or []):
        return ""
    registry = load_capability_registry()
    if registry.status != "valid" or not registry.capabilities:
        return ""
    lines = [
        "# Available Scaffolde capabilities",
        "Use the native `scaffolde_capability` tool for Scaffolde-owned capabilities and PAI accounts. Routing precedence: explicit user override > matching Scaffolde authority/account > generic skills.",
        "If a declared capability fails, describe it as `capability_degraded` rather than claiming the mailbox/service is unavailable.",
    ]
    for cap in registry.capabilities.values():
        operations = ", ".join(f"{name}({op.get('risk')})" for name, op in cap.get("operations", {}).items())
        triggers = ", ".join(cap.get("triggers", [])[:4])
        lines.append(f"- {cap.get('id')} / {cap.get('tool_name')}: {cap.get('description')} Authority: {cap.get('authority')}. Operations: {operations}. Triggers: {triggers}.")
    return "\n".join(lines)
