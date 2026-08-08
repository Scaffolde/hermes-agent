from __future__ import annotations

import pytest

from agent.subagent_tool_boundary import (
    EvoToolBoundaryError,
    classify_evo_tools,
)


def test_classifies_evo_run_as_host_brokered_authority() -> None:
    local, host_brokered = classify_evo_tools(
        ["terminal", "read_file", "scaffolde_evo_agent_dispatch", "scaffolde_evo_run"]
    )

    assert local == {"terminal"}
    assert host_brokered == {
        "read_file",
        "scaffolde_evo_agent_dispatch",
        "scaffolde_evo_run",
    }


def test_rejects_unknown_evo_tool_authority() -> None:
    with pytest.raises(EvoToolBoundaryError, match="undefined"):
        classify_evo_tools(["scaffolde_unknown_tool"])
