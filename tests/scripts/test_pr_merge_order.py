"""Tests for scripts/pr_merge_order.py (SCA-4638).

The load-bearing tests here run against **real git repositories** built in a
tmpdir, not against stub probes. A merge-order sweep is exactly the kind of
tool that can pass a suite of hand-written fakes while answering nothing on a
real queue, and the failure mode this tool exists to catch — "every PR reports
CLEAN and the queue still cannot land" — is only reproducible with actual
three-way merges.

Every positive control is paired with a negative one. A detector that flags
everything would satisfy "it flags the colliding pair" on its own; the paired
test pins that it does *not* flag independent work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pr_merge_order as pmo  # noqa: E402


# ---------------------------------------------------------------------------
# git fixture helpers
# ---------------------------------------------------------------------------


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "commit.gpgsign", "false")
    return repo


def write_commit(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "add", path)
    git(repo, "commit", "--quiet", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


BASE_FILE = "shared.txt"
BASE_CONTENT = "L1\nL2\nL3\nL4\nL5\n"


def base_repo(tmp_path: Path) -> Path:
    """A repo whose ``main`` holds a five-line file every branch will edit."""
    repo = init_repo(tmp_path)
    write_commit(repo, BASE_FILE, BASE_CONTENT, "base")
    return repo


def branch_from(repo: Path, name: str, start: str, path: str, content: str, message: str) -> str:
    git(repo, "checkout", "--quiet", "-B", name, start)
    sha = write_commit(repo, path, content, message)
    git(repo, "checkout", "--quiet", "main")
    return sha


def edited(line_index: int, value: str, content: str = BASE_CONTENT) -> str:
    lines = content.rstrip("\n").split("\n")
    lines[line_index] = value
    return "\n".join(lines) + "\n"


def pr(number: int, head: str, base: str = "main", **kwargs) -> pmo.PullRequest:
    return pmo.PullRequest(number=number, head_ref=head, base_ref=base, **kwargs)


def sweep(repo: Path, prs: list[pmo.PullRequest]) -> pmo.MergeOrderReport:
    suite = pmo.GitProbeSuite(repo, lambda p: p.base_ref if p.base_ref == "main" else "main")
    return pmo.compute_merge_order(prs, suite.probe, suite.is_ancestor, preflight=suite.preflight)


# ---------------------------------------------------------------------------
# AC2 — positive control: a collision GitHub reports as CLEAN
# ---------------------------------------------------------------------------


def test_sibling_collision_is_flagged_though_each_merges_clean_alone(tmp_path: Path):
    """The exact 2026-08-10 shape: two same-base siblings, both CLEAN, un-landable.

    This is the positive control the acceptance criteria demand. Without it the
    sweep could pass by doing nothing — the recurring ``--dry-run`` false-green
    in the ledger. The assertions on ``preflight`` are the "GitHub says CLEAN"
    half: each branch genuinely merges into ``main`` on its own, which is the
    only thing ``mergeStateStatus`` ever measures.
    """
    repo = base_repo(tmp_path)
    branch_from(repo, "feat-a", "main", BASE_FILE, edited(1, "A2"), "a")
    branch_from(repo, "feat-b", "main", BASE_FILE, edited(1, "B2"), "b")

    suite = pmo.GitProbeSuite(repo, "main")
    a, b = pr(1, "feat-a"), pr(2, "feat-b")

    # What GitHub measures, per PR, against its own base: both clean.
    assert suite.preflight(a).conflicts is False
    assert suite.preflight(b).conflicts is False

    report = sweep(repo, [a, b])

    assert [(p.a, p.b) for p in report.mutual_conflicts] == [(1, 2)]
    assert BASE_FILE in report.mutual_conflicts[0].files
    # A mutual conflict is a decision, not an order — neither PR is sequenced.
    assert report.merge_order == []
    assert report.blocked == [1, 2]


def test_independent_siblings_are_not_flagged(tmp_path: Path):
    """Negative control: the detector must be able to say "no collision".

    Paired with the test above so neither can pass by a constant verdict.
    """
    repo = base_repo(tmp_path)
    branch_from(repo, "feat-a", "main", "a.txt", "a\n", "a")
    branch_from(repo, "feat-b", "main", "b.txt", "b\n", "b")

    report = sweep(repo, [pr(1, "feat-a"), pr(2, "feat-b")])

    assert report.mutual_conflicts == []
    assert report.merge_order == [1, 2]
    assert [p.relation for p in report.pairs] == ["independent"]


def test_same_file_different_regions_is_not_a_collision(tmp_path: Path):
    """File overlap is not collision. Both branches rewrite ``shared.txt`` and
    still merge cleanly in both orders — which is why the sweep runs a real
    three-way merge instead of intersecting changed-file lists."""
    repo = base_repo(tmp_path)
    branch_from(repo, "feat-a", "main", BASE_FILE, edited(0, "A1"), "a")
    branch_from(repo, "feat-b", "main", BASE_FILE, edited(4, "B5"), "b")

    report = sweep(repo, [pr(1, "feat-a"), pr(2, "feat-b")])

    assert report.mutual_conflicts == []
    assert report.merge_order == [1, 2]


def test_one_way_collision_yields_an_order_not_a_block():
    """When exactly one direction merges clean, that direction is the answer.

    Driven by a stub probe rather than a git fixture on purpose. A three-way
    merge of two branches off a shared base is symmetric — it conflicts in both
    directions or neither — so no arrangement of same-base siblings can produce
    a one-way verdict to test against. The asymmetric case is real once history
    is non-trivial (a partially cherry-picked branch, a rebased head), and what
    needs pinning is that the sweep turns it into an ordering edge rather than
    a block.
    """
    def probe(a: pmo.PullRequest, b: pmo.PullRequest) -> pmo.ProbeVerdict:
        # #2 conflicts once #1 has landed, so #2 has to land first.
        if (a.number, b.number) == (1, 2):
            return pmo.ProbeVerdict(True, (BASE_FILE,))
        return pmo.ProbeVerdict(False)

    report = pmo.compute_merge_order(
        [pr(1, "feat-a"), pr(2, "feat-b")], probe, lambda _a, _b: False
    )

    ordered = [p for p in report.pairs if p.relation == "ordered"]
    assert len(ordered) == 1
    assert ordered[0].first == 2
    assert report.merge_order == [2, 1]
    assert report.mutual_conflicts == []
    assert report.blocked == []


# ---------------------------------------------------------------------------
# AC3 — merge-commit semantics, and why squash is wrong here
# ---------------------------------------------------------------------------


def squash_land_conflicts(repo: Path, tmp_path: Path, first: str, second: str) -> bool:
    """Simulates landing two branches as SQUASHES, the way ``--squash`` does it.

    A squash landing is the flat diff ``main..branch`` applied as a single
    patch. That is patch application, not a three-way merge, so it has no
    knowledge that the second branch's diff already contains the first's
    commits.
    """
    scratch = tmp_path / f"squash-{first}-{second}"
    subprocess.run(
        ["git", "clone", "--quiet", "--branch", "main", str(repo), str(scratch)],
        check=True, capture_output=True,
    )
    # Each squash diff is taken against the ORIGINAL base, the way a squash
    # merge does it, not against a `main` that earlier squashes have already
    # moved. Diffing against the moving tip would quietly subtract the parent's
    # changes from the child's patch and hide the very collision under test.
    merge_base = git(scratch, "rev-parse", "main").stdout.strip()
    for branch in (first, second):
        diff = git(scratch, "diff", merge_base, f"origin/{branch}").stdout
        patch = tmp_path / f"{branch}.patch"
        patch.write_text(diff, encoding="utf-8")
        applied = git(scratch, "apply", "--index", str(patch), check=False)
        if applied.returncode != 0:
            return True
        git(scratch, "-c", "user.email=t@e.com", "-c", "user.name=T",
            "commit", "--quiet", "-m", f"squash {branch}")
    return False


def test_merge_commit_probe_is_clean_where_a_squash_simulation_false_positives(tmp_path: Path):
    """Pins the merge-commit choice with the evidence, not a comment.

    ``main`` here takes merge commits (40 of the last 40 at the time of
    writing). On a stacked pair the two semantics disagree, and only one of
    them is right: the child's branch already *contains* the parent's commits,
    so a three-way merge sees no new change and lands clean, while a squash
    re-applies the parent's hunks as a flat diff against a base that already
    has them and reports a conflict that does not exist.

    Choosing squash here would have reported a false collision on every stacked
    PR in the queue — which is exactly the discarded first result during the
    2026-08-10 manual sweep.
    """
    repo = base_repo(tmp_path)
    parent_sha = branch_from(repo, "feat-parent", "main", BASE_FILE, edited(1, "P2"), "parent")
    branch_from(repo, "feat-child", parent_sha, BASE_FILE,
                edited(3, "C4", edited(1, "P2")), "child")

    suite = pmo.GitProbeSuite(repo, "main")
    parent, child = pr(1, "feat-parent"), pr(2, "feat-child", base="feat-parent")

    # Merge-commit semantics — what this tool uses. No conflict, correctly.
    assert suite.probe(parent, child).conflicts is False
    # Squash semantics on the identical pair — a conflict that is not real.
    assert squash_land_conflicts(repo, tmp_path, "feat-parent", "feat-child") is True


def test_linear_stack_is_an_ordering_edge_not_a_conflict(tmp_path: Path):
    repo = base_repo(tmp_path)
    parent_sha = branch_from(repo, "feat-parent", "main", BASE_FILE, edited(1, "P2"), "parent")
    branch_from(repo, "feat-child", parent_sha, BASE_FILE,
                edited(3, "C4", edited(1, "P2")), "child")

    report = sweep(repo, [pr(1, "feat-parent"), pr(2, "feat-child", base="feat-parent")])

    assert report.stacked_on == {2: 1}
    assert report.diverged == []
    assert report.merge_order == [1, 2]
    assert report.mutual_conflicts == []


# ---------------------------------------------------------------------------
# shape 2 — the diverged stack that baseRefName renders as linear
# ---------------------------------------------------------------------------


def test_diverged_stack_is_probed_instead_of_skipped(tmp_path: Path):
    """The nastier shape from SCA-4638, and the reason ancestry gates stacking.

    ``#child`` was opened against ``#parent``'s branch, so the PR list renders a
    tidy two-deep stack. Then ``#parent`` was force-pushed somewhere the child
    does not descend from. ``baseRefName`` still says "stacked", and a sweep
    that believed it would mark the pair same-stack and skip the collision
    probe entirely — reporting a landable order for two PRs that collide.

    Only ``git merge-base --is-ancestor`` can tell those apart, so the declared
    link is treated as a claim and the unconfirmed one is reported and probed.
    """
    repo = base_repo(tmp_path)
    parent_sha = branch_from(repo, "feat-parent", "main", BASE_FILE, edited(1, "P2"), "parent v1")
    # Child descends from parent v1 and edits L4.
    branch_from(repo, "feat-child", parent_sha, BASE_FILE,
                edited(3, "C4", edited(1, "P2")), "child")
    # Parent is force-pushed off that history and now also rewrites L4.
    branch_from(repo, "feat-parent", "main", BASE_FILE,
                edited(3, "P4", edited(1, "P2b")), "parent v2 (force-push)")

    parent, child = pr(1, "feat-parent"), pr(2, "feat-child", base="feat-parent")

    suite = pmo.GitProbeSuite(repo, "main")
    # The premise: the declared link is no longer real ancestry.
    assert suite.is_ancestor(parent, child) is False

    report = sweep(repo, [parent, child])

    assert report.diverged == [(2, 1)]
    assert report.stacked_on == {}
    # ...and having been probed rather than skipped, the collision surfaces.
    assert [(p.a, p.b) for p in report.mutual_conflicts] == [(1, 2)]
    assert report.merge_order == []
    assert any("diverged" in e for e in report.errors)


def test_undeclared_ancestry_records_an_ordering_edge(tmp_path: Path):
    """The mirror image: a child targeting ``main`` whose head already contains
    another open PR. Nothing in ``baseRefName`` announces it, but merging the
    child first would land the parent's work under a different PR's review."""
    repo = base_repo(tmp_path)
    parent_sha = branch_from(repo, "feat-parent", "main", BASE_FILE, edited(1, "P2"), "parent")
    branch_from(repo, "feat-child", parent_sha, BASE_FILE,
                edited(3, "C4", edited(1, "P2")), "child")

    # Both PRs declare `main` — no stack is visible on the board.
    report = sweep(repo, [pr(1, "feat-parent"), pr(2, "feat-child")])

    assert report.stacked_on == {2: 1}
    assert any(e.reason == "ancestry" for e in report.edges)
    assert report.merge_order == [1, 2]
    assert any("already contains" in e for e in report.errors)


# ---------------------------------------------------------------------------
# preflight / false-green guards
# ---------------------------------------------------------------------------


def test_pr_conflicting_with_its_own_base_is_held_unverified(tmp_path: Path):
    """A single-PR queue takes every pair skip, so without the preflight the
    report would print a viable order backed by zero probe calls."""
    repo = base_repo(tmp_path)
    branch_from(repo, "feat-a", "main", BASE_FILE, edited(1, "A2"), "a")
    # main moves under it, conflicting.
    write_commit(repo, BASE_FILE, edited(1, "MAIN2"), "main moves")

    report = sweep(repo, [pr(1, "feat-a")])

    assert report.unverified == [1]
    assert report.merge_order == []
    assert any("does not merge cleanly into main" in e for e in report.errors)


def test_unresolvable_head_is_reported_never_silently_dropped(tmp_path: Path):
    repo = base_repo(tmp_path)
    branch_from(repo, "feat-a", "main", "a.txt", "a\n", "a")

    report = sweep(repo, [pr(1, "feat-a"), pr(2, "does-not-exist")])

    assert 2 in report.unverified
    assert 2 not in report.merge_order
    assert any("does-not-exist" in e for e in report.errors)


# ---------------------------------------------------------------------------
# stacking resolution edge cases (pure logic)
# ---------------------------------------------------------------------------


def test_cross_repository_head_named_main_cannot_parent_the_queue():
    """A fork PR whose head branch is literally ``main`` has
    ``head_ref == "main"``, which equals the base of every ordinary PR. Matching
    on the name alone would declare the entire queue stacked on it and switch
    off every collision probe. This repo currently has such a PR open."""
    fork_pr = pr(99, "main", base="main", cross_repository=True)
    prs = [fork_pr, pr(1, "feat-a"), pr(2, "feat-b")]

    stacked, diverged = pmo.resolve_stacking(prs, lambda _a, _b: True)

    assert stacked == {}
    assert diverged == []


def test_stack_ancestors_is_cycle_safe():
    assert pmo.stack_ancestors(1, {1: 2, 2: 1}) == [2]


def test_topological_order_breaks_ties_by_number():
    order, cyclic = pmo.topological_order(
        [3, 1, 2], [pmo.OrderingEdge(frm=3, to=1, reason="stacked")]
    )
    assert order == [2, 3, 1]
    assert cyclic == []


def test_topological_order_reports_a_cycle_instead_of_dropping_nodes():
    order, cyclic = pmo.topological_order(
        [1, 2],
        [pmo.OrderingEdge(frm=1, to=2, reason="conflict"),
         pmo.OrderingEdge(frm=2, to=1, reason="conflict")],
    )
    assert order == []
    assert cyclic == [1, 2]


def test_separate_trunks_are_never_probed_against_each_other(tmp_path: Path):
    repo = base_repo(tmp_path)
    git(repo, "branch", "release")
    branch_from(repo, "feat-a", "main", BASE_FILE, edited(1, "A2"), "a")
    branch_from(repo, "feat-b", "main", BASE_FILE, edited(1, "B2"), "b")

    suite = pmo.GitProbeSuite(repo, "main")
    report = pmo.compute_merge_order(
        [pr(1, "feat-a"), pr(2, "feat-b", base="release")],
        suite.probe, suite.is_ancestor, preflight=suite.preflight,
    )

    assert [p.relation for p in report.pairs] == ["separate-base"]
    assert report.mutual_conflicts == []


# ---------------------------------------------------------------------------
# queue discovery — repo pinning and the empty-queue fail-open
# ---------------------------------------------------------------------------


def test_repo_slug_is_pinned_to_a_named_remote(tmp_path: Path):
    """This checkout carries ``origin``/``pai-scaffolde`` (the fork) and
    ``upstream`` (NousResearch). With no default set, ``gh pr list`` resolves to
    *upstream* and returns its ~85,000-PR queue — a sweep that trusted that
    would report an order for PRs this repository never lands."""
    repo = init_repo(tmp_path)
    git(repo, "remote", "add", "origin", "https://github.com/pai-scaffolde/hermes-agent")
    git(repo, "remote", "add", "upstream", "https://github.com/NousResearch/hermes-agent")

    assert pmo.resolve_repo_slug(repo, "origin") == "pai-scaffolde/hermes-agent"
    assert pmo.resolve_repo_slug(repo, "upstream") == "NousResearch/hermes-agent"


def test_repo_slug_parses_ssh_and_dot_git_forms(tmp_path: Path):
    repo = init_repo(tmp_path)
    git(repo, "remote", "add", "origin", "git@github.com:pai-scaffolde/hermes-agent.git")
    assert pmo.resolve_repo_slug(repo, "origin") == "pai-scaffolde/hermes-agent"


def test_empty_queue_must_be_corroborated_by_a_second_code_path():
    """``gh pr list`` exiting 0 with ``[]`` is indistinguishable from a query
    that answered nothing. Believing it is a fail-open: the sweep goes green
    having examined zero PRs."""
    with pytest.raises(RuntimeError, match="refusing to report a swept queue"):
        pmo.assert_queue_genuinely_empty(lambda: 4)

    pmo.assert_queue_genuinely_empty(lambda: 0)  # agreement is the only pass


def test_fetch_args_refresh_every_base_and_never_write_fetch_head():
    prs = [pr(1, "feat-a"), pr(2, "feat-b", base="release")]
    args = pmo.build_fetch_args(prs, "origin")

    assert "--no-write-fetch-head" in args
    assert "+refs/pull/1/head:refs/pr-merge-order/1" in args
    # The bases must be refreshed too, or every pair is probed against an
    # obsolete trunk that nobody is merging into.
    assert "+refs/heads/main:refs/remotes/origin/main" in args
    assert "+refs/heads/release:refs/remotes/origin/release" in args


def test_prune_drops_only_this_tools_refs(tmp_path: Path):
    repo = base_repo(tmp_path)
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    git(repo, "update-ref", "refs/pr-merge-order/7", head)
    git(repo, "update-ref", "refs/pr-merge-order/8", head)
    git(repo, "update-ref", "refs/heads/keep-me", head)

    pruned = pmo.prune_probe_refs(repo, [7])

    assert pruned == ["refs/pr-merge-order/8"]
    assert git(repo, "rev-parse", "--verify", "refs/pr-merge-order/7").returncode == 0
    assert git(repo, "rev-parse", "--verify", "refs/heads/keep-me").returncode == 0


# ---------------------------------------------------------------------------
# wiring — a detector nothing listens to is not a fix (SCA-4633)
# ---------------------------------------------------------------------------


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-merge-order.yml"


def test_sweep_is_scheduled_and_runs_the_script():
    assert WORKFLOW.is_file(), "the sweep has no workflow; nothing would ever run it"
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "schedule:" in text and "cron:" in text
    assert "workflow_dispatch:" in text
    assert "scripts/pr_merge_order.py" in text


def test_sweep_checks_out_full_history():
    """merge-base/merge-tree need real history. Under a shallow clone every
    ancestry answer is wrong, which silently disables the stack-divergence
    check — the sweep would still exit 0 and report a landable order."""
    assert "fetch-depth: 0" in WORKFLOW.read_text(encoding="utf-8")


def test_collision_verdict_reaches_a_human():
    """AC4. The exit-1 path must open or update a GitHub issue and carry the
    report with it, not merely colour a check red on a page nobody opens."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "steps.sweep.outputs.status == '1'" in text
    assert "gh issue create" in text
    assert "gh issue comment" in text  # append rather than spam a new issue each tick
    assert "issues: write" in text
    assert "GITHUB_STEP_SUMMARY" in text


def test_sweep_failure_is_not_reported_as_a_clean_queue():
    """Exit 2 means the sweep could not run. A broken detector must fail the
    job: treating "could not verify" as "verified" is the false-green this
    whole tool exists to end."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'status" -eq 2' in text
    assert "queue was NOT verified" in text


def test_sweep_is_pinned_to_the_fork_queue():
    """Upstream carries tens of thousands of open PRs; an all-pairs sweep there
    is meaningless for this repository and enormously expensive."""
    assert "github.repository == 'pai-scaffolde/hermes-agent'" in WORKFLOW.read_text(encoding="utf-8")


def test_parse_merge_tree_output_separates_paths_from_prose():
    tree, files = pmo.parse_merge_tree_output(
        "abc123\nshared.txt\nother.txt\n\nCONFLICT (content): Merge conflict in shared.txt\n"
    )
    assert tree == "abc123"
    assert files == ["shared.txt", "other.txt"]

    tree, files = pmo.parse_merge_tree_output("abc123\n")
    assert (tree, files) == ("abc123", [])
