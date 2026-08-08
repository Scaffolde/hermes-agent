"""Exact Evo v1 tool classification shared by host and owned worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

LOCAL_EVO_TOOL_NAMES = frozenset({"terminal", "write_file", "patch", "search_files"})
HOST_BROKERED_EVO_TOOL_NAMES = frozenset({
    "read_file",
    "scaffolde_evo_agent_dispatch",
})
SUPPORTED_EVO_TOOL_NAMES = LOCAL_EVO_TOOL_NAMES | HOST_BROKERED_EVO_TOOL_NAMES


class EvoToolBoundaryError(ValueError):
    """An exact process profile requested a tool outside Evo v1 authority."""


def classify_evo_tools(names: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    exact = frozenset(names)
    unknown = exact - SUPPORTED_EVO_TOOL_NAMES
    if unknown:
        raise EvoToolBoundaryError(
            f"Evo v1 tool classification is undefined for: {sorted(unknown)}"
        )
    return exact & LOCAL_EVO_TOOL_NAMES, exact & HOST_BROKERED_EVO_TOOL_NAMES


def exact_tool_schema_digest(tools: list[Mapping[str, Any]]) -> str:
    functions: list[Mapping[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        functions.append(function if isinstance(function, Mapping) else tool)
    encoded = json.dumps(
        sorted(functions, key=lambda entry: str(entry.get("name", ""))),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
