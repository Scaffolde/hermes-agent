from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/upstream-sync-pr.yml"


def test_scaffolde_stable_release_sync_workflow_is_preserved() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.repository == 'pai-scaffolde/hermes-agent'" in text
    assert "repos/NousResearch/hermes-agent/releases/latest" in text
    assert 'git merge --no-ff --no-edit "$LATEST_SHA"' in text
    assert "git rebase" not in text
    assert 'gh workflow run ci.yml' in text
    assert "--merge" in text
    assert "--delete-branch" in text
