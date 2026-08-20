#!/usr/bin/env python3
"""pr_merge_order — compute a landable merge order for the open PR queue.

GitHub's ``mergeStateStatus`` is a per-PR measurement against that PR's own
base. It is never a measurement of the *queue*. Two collision shapes are
therefore structurally invisible on the board, and both cost us a real stall on
2026-08-10 when six PRs all read ``CLEAN`` and the queue could not land in any
order (SCA-4638):

1. **Same-base siblings** — two PRs both based on ``main`` touching the same
   file. Each is compared to ``main``, never to the other.
2. **Diverged stack siblings** — two PRs that declare the same open PR as their
   base but do not both actually descend from its current head. The PR list
   renders this as a tidy linear stack; only ``git merge-base --is-ancestor``
   reveals the fork.

Shape 2 is the nastier one, and it is why this tool refuses to trust
``baseRefName``. A declared stack link is a *claim*; it is honoured as a hard
ordering edge (and as a reason to skip the collision probe) only when real
ancestry confirms it. An unconfirmed link is a divergence, and the pair gets
probed like any other siblings — see :func:`resolve_stacking`.

File overlap is not a usable proxy for collision either: two branches can
rewrite the same 500 lines and still merge clean both ways. Only a real
three-way merge answers it, so each ordered pair is probed with
``git merge-tree --write-tree``: land A onto the base as a synthetic commit,
then merge B onto that result.

**Merge-commit semantics, deliberately.** ``merge-tree`` is a true three-way
merge, which is what this repository actually does (the last 40 commits on
``main`` are all merge commits). Simulating a *squash* instead reports false
collisions on every stacked PR, because the child re-applies its parent's
changes as a flat diff against a base that already contains them. That false
positive is pinned by a test rather than left as a comment.

Read-only with respect to the queue and the working tree: no checkout, no
merge, no branch write. PR heads are fetched into this tool's own
``refs/pr-merge-order/<n>`` namespace, and ``merge-tree`` writes only loose
objects, so it stays safe to run while the queue is frozen.

Usage::

    python3 scripts/pr_merge_order.py            # human-readable report
    python3 scripts/pr_merge_order.py --json     # machine-readable report

Exit codes are derived from *verified coverage*, never from "no conflict was
recorded" — a sweep that measured nothing must not read like a clean one:

* ``0`` — every open PR was measured and every one of them is in the order.
* ``1`` — every open PR was measured and some of them have no landing order
  (a mutual conflict, an ordering cycle, or a descendant of either).
* ``2`` — verification is incomplete: a PR got no verdict, the queue was
  truncated by ``--limit``, or the sweep could not run at all.

The workflow treats any status outside this set as a failure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

PROBE_REF_NAMESPACE = "refs/pr-merge-order/"


@dataclass(frozen=True)
class PullRequest:
    number: int
    head_ref: str
    base_ref: str
    title: str = ""
    #: True when the head branch lives in a fork rather than this repository.
    cross_repository: bool = False


@dataclass(frozen=True)
class ProbeVerdict:
    #: True when merging ``b`` onto (base + ``a``) hits a conflict.
    conflicts: bool
    #: Conflicted paths, when the probe can name them.
    files: tuple[str, ...] = ()


@dataclass
class PairFinding:
    a: int
    b: int
    #: independent | ordered | mutual | separate-base
    relation: str
    #: For ``ordered``: the PR that must land first.
    first: int | None = None
    files: list[str] = field(default_factory=list)


@dataclass
class OrderingEdge:
    #: Must land before ``to``.
    frm: int
    to: int
    #: stacked | ancestry | conflict
    reason: str


@dataclass
class MergeOrderReport:
    prs: list[int] = field(default_factory=list)
    #: PR -> the PR its branch actually sits on top of (ancestry-confirmed).
    stacked_on: dict[int, int] = field(default_factory=dict)
    #: (child, declared parent) links that ancestry did NOT confirm.
    diverged: list[tuple[int, int]] = field(default_factory=list)
    #: PR -> the trunk its stack ultimately lands on. Only same-trunk PRs compare.
    bases: dict[int, str] = field(default_factory=dict)
    pairs: list[PairFinding] = field(default_factory=list)
    edges: list[OrderingEdge] = field(default_factory=list)
    #: Topologically viable landing order. Every entry carries a real verdict.
    merge_order: list[int] = field(default_factory=list)
    #: Pairs that conflict in BOTH directions — one must be rebased onto the other.
    mutual_conflicts: list[PairFinding] = field(default_factory=list)
    #: PRs left out of ``merge_order`` because something must be resolved first.
    blocked: list[int] = field(default_factory=list)
    #: PRs held out because no probe ever returned a verdict for them. Distinct
    #: from ``blocked``: those have an answer, these have none.
    unverified: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prs": self.prs,
            "stackedOn": {str(k): v for k, v in self.stacked_on.items()},
            "diverged": [{"child": c, "declaredParent": p} for c, p in self.diverged],
            "bases": {str(k): v for k, v in self.bases.items()},
            "pairs": [
                {"a": p.a, "b": p.b, "relation": p.relation, "first": p.first, "files": p.files}
                for p in self.pairs
            ],
            "edges": [{"from": e.frm, "to": e.to, "reason": e.reason} for e in self.edges],
            "mergeOrder": self.merge_order,
            "mutualConflicts": [
                {"a": p.a, "b": p.b, "files": p.files} for p in self.mutual_conflicts
            ],
            "blocked": self.blocked,
            "unverified": self.unverified,
            "errors": self.errors,
        }


#: Probes "after ``a`` lands, does ``b`` still merge cleanly?".
MergeProbe = Callable[[PullRequest, PullRequest], ProbeVerdict]
#: Checks a single PR against its own base, independently of any sibling.
BasePreflight = Callable[[PullRequest], ProbeVerdict]
#: True when ``ancestor``'s head commit is an ancestor of ``descendant``'s head.
AncestryProbe = Callable[[PullRequest, PullRequest], bool]


# ---------------------------------------------------------------------------
# ancestry / stacking
# ---------------------------------------------------------------------------


def declared_parents(prs: Iterable[PullRequest]) -> dict[int, int]:
    """PR -> the open PR its ``base_ref`` names, ancestry notwithstanding.

    This is the *claim*, used only for trunk attribution — a PR opened against
    another PR's branch belongs to that PR's queue whether or not the stack has
    since diverged. Ordering decisions use :func:`resolve_stacking` instead,
    which confirms the claim against real ancestry.

    Only a SAME-REPOSITORY head can be a parent. Branch names are not globally
    unique: a PR opened from a fork branch named ``main`` has
    ``head_ref == "main"``, which equals the base of every ordinary PR in the
    queue. Matching on the name alone would declare the whole queue stacked on
    that one PR and switch off every collision probe — this tool's entire job,
    disabled by a branch someone else happened to name ``main``.
    """
    prs = list(prs)
    by_head: dict[str, int] = {}
    for pr in prs:
        if pr.cross_repository:
            continue
        by_head[pr.head_ref] = pr.number

    parents: dict[int, int] = {}
    for pr in prs:
        parent = by_head.get(pr.base_ref)
        if parent is not None and parent != pr.number:
            parents[pr.number] = parent
    return parents


def resolve_stacking(
    prs: Iterable[PullRequest],
    is_ancestor: AncestryProbe,
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Maps each PR to the open PR it actually sits on top of.

    Returns ``(stacked_on, diverged)``.

    A stacked PR is a hard ordering edge, not a conflict: without recognising
    it, the parent's own commits read as a collision with the child.

    ``baseRefName`` alone cannot establish that relationship. It records what
    the branch was *opened* against, and it keeps saying so after a rebase,
    reset, or force-push moves the child off its parent's current head. Trusting
    it means marking a diverged pair "same stack" and skipping the collision
    probe — the exact blind spot that let #62/#67-shaped siblings read as a
    linear stack. Every declared link is therefore confirmed against real
    ancestry, and an unconfirmed one is recorded in ``diverged`` and dropped so
    the pair gets probed.

    Only a SAME-REPOSITORY head can be a stack parent. Branch names are not
    globally unique: a PR opened from a fork branch named ``main`` has
    ``head_ref == "main"``, which equals the base of every ordinary PR in the
    queue. Matching on the name alone would declare the whole queue stacked on
    that one PR and switch off every collision probe — this tool's entire job,
    disabled by a branch someone else happened to name ``main``.
    """
    prs = list(prs)
    by_number = {pr.number: pr for pr in prs}

    stacked_on: dict[int, int] = {}
    diverged: list[tuple[int, int]] = []
    for child, parent in declared_parents(prs).items():
        if is_ancestor(by_number[parent], by_number[child]):
            stacked_on[child] = parent
        else:
            diverged.append((child, parent))
    return stacked_on, diverged


def stack_ancestors(pr: int, stacked_on: dict[int, int]) -> list[int]:
    """Every PR beneath ``pr`` in the stack, nearest first. Cycle-safe."""
    chain: list[int] = []
    seen = {pr}
    cursor = stacked_on.get(pr)
    while cursor is not None and cursor not in seen:
        chain.append(cursor)
        seen.add(cursor)
        cursor = stacked_on.get(cursor)
    return chain


def _same_stack(a: int, b: int, stacked_on: dict[int, int]) -> bool:
    return b in stack_ancestors(a, stacked_on) or a in stack_ancestors(b, stacked_on)


def ancestor_closure(pr: int, parents: dict[int, set[int]]) -> set[int]:
    """Every PR reachable beneath ``pr`` across the full ancestry DAG.

    ``stacked_on`` is a scalar chain and can only remember ONE parent, but a
    head that merged two feature branches genuinely contains two open PRs.
    Propagating blocked state along the scalar chain alone lets the survivor of
    that overwrite escape: the descendant is skipped by preflight as a
    non-root, yet nothing holds it back when the forgotten ancestor is the one
    that cannot land — so it reaches ``merge_order`` still carrying that
    ancestor's unmergeable commits. Cycle-safe.
    """
    seen: set[int] = set()
    frontier = list(parents.get(pr, ()))
    while frontier:
        cursor = frontier.pop()
        if cursor in seen or cursor == pr:
            continue
        seen.add(cursor)
        frontier.extend(parents.get(cursor, ()))
    return seen


def effective_base_ref(
    pr: int,
    by_number: dict[int, PullRequest],
    declared: dict[int, int],
) -> str:
    """The trunk a PR's stack ultimately lands on.

    Two PRs are only comparable when they land on the same history. A PR
    targeting a release or maintenance branch shows up in ``gh pr list``
    alongside the ``main`` queue, and probing it against ``main`` would compare
    it to a history it never merges into — inventing conflicts, or clearing
    real ones.

    Attribution walks the DECLARED parent chain, not the ancestry-confirmed
    one. A child whose stack has diverged still lands on its parent's trunk;
    resolving it from its own ``base_ref`` would read that parent's branch name
    as a separate trunk and file the pair as ``separate-base`` — silently
    dropping the exact collision the divergence check just exposed.
    """
    seen = {pr}
    root = pr
    while root in declared and declared[root] not in seen:
        root = declared[root]
        seen.add(root)
    found = by_number.get(root)
    return found.base_ref if found else ""


def topological_order(
    nodes: list[int],
    edges: list[OrderingEdge],
) -> tuple[list[int], list[int]]:
    """Kahn topological sort. Ties break by PR number, so the order is stable
    across runs — an order that shuffles is not reviewable."""
    indegree = {n: 0 for n in nodes}
    outgoing: dict[int, list[int]] = {n: [] for n in nodes}
    for edge in edges:
        if edge.frm not in indegree or edge.to not in indegree:
            continue
        outgoing[edge.frm].append(edge.to)
        indegree[edge.to] += 1

    ready = sorted(n for n in nodes if indegree[n] == 0)
    order: list[int] = []
    while ready:
        nxt = ready.pop(0)
        order.append(nxt)
        for target in outgoing[nxt]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()

    cyclic = [n for n in nodes if n not in order]
    return order, cyclic


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def compute_merge_order(
    prs: Iterable[PullRequest],
    probe: MergeProbe,
    is_ancestor: AncestryProbe,
    preflight: BasePreflight | None = None,
    replay: Callable[[list[PullRequest]], int | None] | None = None,
) -> MergeOrderReport:
    """Classifies every candidate pair and derives a viable landing order.

    ``replay`` receives a candidate sequence sharing one trunk and returns the
    number of the first PR that fails to merge onto the accumulated result, or
    ``None`` when the whole sequence lands. Without it the emitted order is
    only pairwise-verified; see the replay block below for why that is weaker.
    """
    prs = list(prs)
    numbers = sorted(pr.number for pr in prs)
    by_number = {pr.number: pr for pr in prs}
    stacked_on, diverged = resolve_stacking(prs, is_ancestor)
    declared = declared_parents(prs)

    report = MergeOrderReport(prs=numbers, stacked_on=stacked_on, diverged=diverged)
    #: No verdict could be established (a probe raised). Drives exit 2.
    unverified: set[int] = set()
    #: Measured as unable to merge into its own base. A real verdict, so it
    #: blocks the PR without claiming the sweep failed.
    unmergeable: set[int] = set()

    for number in numbers:
        report.bases[number] = effective_base_ref(number, by_number, declared)

    for child, parent in stacked_on.items():
        report.edges.append(OrderingEdge(frm=parent, to=child, reason="stacked"))

    for child, parent in diverged:
        report.errors.append(
            f"#{child} declares #{parent} as its base but does not descend from its head — "
            "the stack has diverged; probing them as siblings"
        )

    # An undeclared stack is the mirror image: the child targets the trunk, yet
    # its head already contains the parent's commits. Merging the child first
    # would silently land the parent's work under a different PR's review, so
    # the ordering edge is recorded even though no base ref announces it.
    # Every confirmed ancestry link, not just the last one written to the
    # scalar chain. Blocked-state propagation reads this DAG.
    ancestry: dict[int, set[int]] = {}
    for child, parent in stacked_on.items():
        ancestry.setdefault(child, set()).add(parent)

    for a, b in itertools.permutations(numbers, 2):
        if stacked_on.get(b) == a or _same_stack(a, b, stacked_on):
            continue
        if is_ancestor(by_number[a], by_number[b]):
            ancestry.setdefault(b, set()).add(a)
            stacked_on[b] = a
            report.stacked_on[b] = a
            report.edges.append(OrderingEdge(frm=a, to=b, reason="ancestry"))
            report.errors.append(
                f"#{b} does not declare a base of #{a}, but its head already contains "
                f"#{a} — landing #{b} first would merge #{a} with it"
            )

    # Every stack ROOT is checked against its own base before any pair is
    # considered. Without this a single-PR queue, or a purely linear stack,
    # takes the same-stack skip on every possible pair, reaches the report with
    # zero probe calls behind it, and prints a viable order at exit 0 — the
    # exact false-green this tool exists to end.
    if preflight is not None:
        for number in numbers:
            if number in stacked_on:
                continue  # not a root; its root is checked instead
            pr = by_number[number]
            try:
                verdict = preflight(pr)
            except Exception as error:  # noqa: BLE001 - reported, never swallowed
                unverified.add(number)
                report.errors.append(
                    f"preflight #{number} against {pr.base_ref} failed: {error}"
                )
                continue
            if verdict.conflicts:
                # A measured verdict, NOT a verification failure. The PR was
                # examined and the answer is "this cannot land at all" — the
                # board shows it DIRTY. Filing it under `unverified` would make
                # the sweep report that it could not verify the queue (exit 2)
                # on every run while any DIRTY PR is open, going red for a
                # working detector and suppressing the collision issue, which
                # only opens on a real exit-1 verdict.
                detail = f" ({', '.join(verdict.files)})" if verdict.files else ""
                unmergeable.add(number)
                report.errors.append(
                    f"#{number} does not merge cleanly into {pr.base_ref} on its own{detail}"
                )

    for a_num, b_num in itertools.combinations(numbers, 2):
        a, b = by_number[a_num], by_number[b_num]
        if _same_stack(a_num, b_num, stacked_on):
            continue

        # Different trunks are different queues; a cross-base verdict would be
        # measured against a history neither PR lands on.
        if report.bases[a_num] != report.bases[b_num]:
            report.pairs.append(PairFinding(a=a_num, b=b_num, relation="separate-base"))
            continue

        # A PR with no usable verdict of its own cannot lend one to a pair:
        # merging onto a branch that does not merge into its own base answers
        # nothing, whether that is because it was never measured (unverified)
        # or because it was measured and cannot land (unmergeable).
        if a_num in unverified or b_num in unverified:
            continue
        if a_num in unmergeable or b_num in unmergeable:
            continue

        try:
            forward = probe(a, b)
            backward = probe(b, a)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            report.errors.append(f"probe #{a_num} <-> #{b_num} failed: {error}")
            # Neither PR has a verdict for this pair, so neither can be placed
            # in an order that claims to be collision-free.
            unverified.add(a_num)
            unverified.add(b_num)
            continue

        files = sorted(set(forward.files) | set(backward.files))

        if forward.conflicts and backward.conflicts:
            finding = PairFinding(a=a_num, b=b_num, relation="mutual", files=files)
            report.pairs.append(finding)
            report.mutual_conflicts.append(finding)
        elif forward.conflicts or backward.conflicts:
            # The direction that merges clean is the viable one: if b conflicts
            # after a lands, then b has to land first.
            first = b_num if forward.conflicts else a_num
            second = a_num if first == b_num else b_num
            report.pairs.append(
                PairFinding(a=a_num, b=b_num, relation="ordered", first=first, files=files)
            )
            report.edges.append(OrderingEdge(frm=first, to=second, reason="conflict"))
        else:
            report.pairs.append(PairFinding(a=a_num, b=b_num, relation="independent"))

    # A mutual conflict is a decision, not an order. Hold both PRs out rather
    # than emitting an order that cannot actually land.
    blocked: set[int] = set(unverified) | set(unmergeable)
    for pair in report.mutual_conflicts:
        blocked.add(pair.a)
        blocked.add(pair.b)
    # Anything stacked on a held PR cannot land either.
    for number in numbers:
        if any(anc in blocked for anc in ancestor_closure(number, ancestry)):
            blocked.add(number)

    orderable = [n for n in numbers if n not in blocked]
    order, cyclic = topological_order(
        orderable,
        [e for e in report.edges if e.frm in orderable and e.to in orderable],
    )
    for n in cyclic:
        blocked.add(n)
        report.errors.append(f"#{n} sits in an ordering cycle and cannot be sequenced automatically")

    # Pairwise-clean does not imply the sequence lands. Git merges are not
    # associative: three branches can be clean in all three pairs and still
    # conflict when the third is merged onto the result of the first two,
    # because the pair probe never showed it the other's changes. Every probe
    # above starts from the trunk plus ONE landed PR, so the emitted order is a
    # hypothesis until it is replayed against a cumulatively advancing base.
    #
    # On failure the culprit is blocked and the order recomputed, which is a
    # bounded search for another topological order rather than a bare refusal.
    # Each pass blocks at least one PR, so the loop terminates.
    if replay is not None:
        for _attempt in range(len(numbers) + 1):
            if not order:
                break
            culprit = _first_replay_failure(order, by_number, report.bases, replay)
            if culprit is None:
                break
            blocked.add(culprit)
            report.errors.append(
                f"#{culprit} merges cleanly against every PR individually but conflicts "
                "when replayed onto the accumulated result of the ones before it — "
                "pairwise-clean is not sequence-clean"
            )
            for number in numbers:
                if any(anc in blocked for anc in ancestor_closure(number, ancestry)):
                    blocked.add(number)
            orderable = [n for n in numbers if n not in blocked]
            order, cyclic = topological_order(
                orderable,
                [e for e in report.edges if e.frm in orderable and e.to in orderable],
            )
            for n in cyclic:
                blocked.add(n)
                report.errors.append(
                    f"#{n} sits in an ordering cycle and cannot be sequenced automatically"
                )

    report.merge_order = order
    report.blocked = sorted(blocked)
    report.unverified = sorted(unverified)
    return report


def _first_replay_failure(
    order: list[int],
    by_number: dict[int, PullRequest],
    bases: dict[int, str],
    replay: Callable[[list[PullRequest]], int | None],
) -> int | None:
    """Replays ``order`` per trunk and returns the first PR that will not land.

    PRs targeting different trunks are different queues — replaying them into
    one sequence would merge histories that never meet in reality — so each
    trunk is replayed on its own, in the relative order the sort produced.
    """
    by_trunk: dict[str, list[PullRequest]] = {}
    for number in order:
        by_trunk.setdefault(bases.get(number, ""), []).append(by_number[number])
    for sequence in by_trunk.values():
        if len(sequence) < 2:
            continue  # a single PR onto its own trunk is exactly the preflight
        culprit = replay(sequence)
        if culprit is not None:
            return culprit
    return None


#: Exit statuses. The workflow rejects anything outside this set, so a crashed
#: interpreter (127) or a killed process (137) can never read as a verdict.
EXIT_LANDABLE = 0
EXIT_NO_ORDER = 1
EXIT_UNVERIFIED = 2


def exit_code_for(report: MergeOrderReport) -> int:
    """Derives the process exit status from *verified coverage*.

    The rule is that a status is a claim about the whole queue, so it may only
    be issued when the whole queue was actually measured:

    * ``2`` — verification is incomplete. At least one PR never got a verdict,
      so nothing can be said about the queue. This is the status that stops a
      broken detector from reading as a clean one: when a fresh checkout could
      not synthesize probe commits, EVERY pair failed, and the old expression
      (``mutual_conflicts or diverged``) still returned 0 — announcing a
      landable queue on zero measurements.
    * ``1`` — every PR was measured and the constraints admit no landing order
      for some of them (a mutual conflict, an ordering cycle, or a descendant
      held back by either).
    * ``0`` — every PR was measured and every one of them is in ``merge_order``.

    ``diverged`` is deliberately NOT a status of its own. A diverged stack is
    re-probed as a sibling pair, so it either produces real verdicts (and is
    covered here like anything else) or fails to (and lands in ``unverified``).
    Failing on the divergence itself reported an un-landable queue for stacks
    the pair probes had already sequenced cleanly.
    """
    if report.unverified:
        return EXIT_UNVERIFIED
    if set(report.merge_order) != set(report.prs):
        return EXIT_NO_ORDER
    return EXIT_LANDABLE


# ---------------------------------------------------------------------------
# git probe
# ---------------------------------------------------------------------------


#: Identity for the disposable probe commits. ``git commit-tree`` demands an
#: author and a committer, and on a fresh CI checkout there is no ``user.name``
#: or ``user.email`` to fall back on: git auto-derives ``runner@host.(none)``,
#: rejects it as bogus under IDENT_STRICT, and exits 128 "Author identity
#: unknown". Every pair probe would then fail, and — before the exit contract
#: below gained teeth — the sweep reported a landable queue having merged
#: nothing. These commits are dangling objects that are never pushed, so a
#: fixed synthetic identity is the honest label, not a placeholder.
PROBE_IDENTITY = {
    "GIT_AUTHOR_NAME": "pr-merge-order",
    "GIT_AUTHOR_EMAIL": "pr-merge-order@invalid",
    "GIT_COMMITTER_NAME": "pr-merge-order",
    "GIT_COMMITTER_EMAIL": "pr-merge-order@invalid",
}


def _git(
    repo_root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **env} if env else None,
    )


def rev_parse_commit(repo_root: Path, ref: str) -> str:
    """Resolves a ref to a commit SHA, or returns ``""``.

    ``git rev-parse <bad-ref>`` echoes the ref back on stdout and exits
    non-zero, so truthy stdout is not proof of resolution. Reading it as one
    turns an unresolvable head into a false CONFLICT verdict.
    """
    result = _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def parse_merge_tree_output(stdout: str) -> tuple[str, list[str]]:
    """Parses ``git merge-tree --write-tree --name-only`` output.

    Layout: tree OID on line 1, then the conflicted paths, then a blank line,
    then human-readable messages. A clean merge emits the OID alone.
    """
    block = re.split(r"\n\s*\n", stdout, maxsplit=1)[0]
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return "", []
    return lines[0], lines[1:]


class GitProbeSuite:
    """Builds merge/ancestry probes against a real repository.

    ``probe`` merges ``a`` onto its trunk into a synthetic commit, then merges
    ``b`` onto that. Nothing is checked out and no ref is written; the synthetic
    commit is a dangling object the next ``git gc`` collects.
    """

    def __init__(self, repo_root: Path, base_ref_for: Callable[[PullRequest], str] | str):
        self.repo_root = repo_root
        if isinstance(base_ref_for, str):
            self._base_ref_of: Callable[[PullRequest], str] = lambda _pr: base_ref_for
        else:
            self._base_ref_of = base_ref_for
        self._base_sha_cache: dict[str, str] = {}
        self._landed_cache: dict[str, str] = {}

    def _base_sha(self, pr: PullRequest) -> tuple[str, str]:
        ref = self._base_ref_of(pr)
        cached = self._base_sha_cache.get(ref)
        if cached:
            return ref, cached
        sha = rev_parse_commit(self.repo_root, ref)
        if not sha:
            raise RuntimeError(f"cannot resolve base ref {ref}")
        self._base_sha_cache[ref] = sha
        return ref, sha

    def _merge_onto_base(self, pr: PullRequest) -> tuple[ProbeVerdict, str, str]:
        _ref, base_sha = self._base_sha(pr)
        head = rev_parse_commit(self.repo_root, pr.head_ref)
        if not head:
            raise RuntimeError(f"cannot resolve head ref {pr.head_ref} (#{pr.number})")
        merged = _git(
            self.repo_root, "merge-tree", "--write-tree", "--name-only", base_sha, head
        )
        tree, files = parse_merge_tree_output(merged.stdout)
        conflicts = merged.returncode != 0 or not tree
        return ProbeVerdict(conflicts, tuple(files) if conflicts else ()), tree, head

    def preflight(self, pr: PullRequest) -> ProbeVerdict:
        return self._merge_onto_base(pr)[0]

    def _land(self, pr: PullRequest) -> str:
        cached = self._landed_cache.get(pr.head_ref)
        if cached:
            return cached
        ref, base_sha = self._base_sha(pr)
        verdict, tree, head = self._merge_onto_base(pr)
        if verdict.conflicts:
            raise RuntimeError(f"#{pr.number} does not merge cleanly into {ref} on its own")
        result = _git(
            self.repo_root,
            "commit-tree", tree, "-p", base_sha, "-p", head,
            "-m", f"probe: land #{pr.number}",
            env=PROBE_IDENTITY,
        )
        commit = result.stdout.strip()
        if not commit:
            # Carry git's own words. "failed to synthesize" alone sent an
            # earlier investigation looking at the merge instead of at the
            # identity that actually rejected the commit.
            detail = result.stderr.strip() or f"exit {result.returncode}"
            raise RuntimeError(
                f"failed to synthesize landing commit for #{pr.number}: {detail}"
            )
        self._landed_cache[pr.head_ref] = commit
        return commit

    def probe(self, a: PullRequest, b: PullRequest) -> ProbeVerdict:
        landed_a = self._land(a)
        head_b = rev_parse_commit(self.repo_root, b.head_ref)
        if not head_b:
            raise RuntimeError(f"cannot resolve head ref {b.head_ref} (#{b.number})")
        result = _git(
            self.repo_root, "merge-tree", "--write-tree", "--name-only", landed_a, head_b
        )
        _tree, files = parse_merge_tree_output(result.stdout)
        conflicts = result.returncode != 0
        return ProbeVerdict(conflicts, tuple(files) if conflicts else ())

    def replay(self, sequence: list[PullRequest]) -> int | None:
        """Merges ``sequence`` onto a cumulatively advancing base.

        This is the difference between "every pair is clean" and "this order
        lands". Each step merges the next head onto the synthetic commit that
        represents everything already landed, exactly as the real queue would.
        Returns the first PR that conflicts, or ``None`` if all of them land.
        """
        if not sequence:
            return None
        _ref, accumulated = self._base_sha(sequence[0])
        for pr in sequence:
            head = rev_parse_commit(self.repo_root, pr.head_ref)
            if not head:
                raise RuntimeError(f"cannot resolve head ref {pr.head_ref} (#{pr.number})")
            merged = _git(
                self.repo_root,
                "merge-tree", "--write-tree", "--name-only", accumulated, head,
            )
            tree, _files = parse_merge_tree_output(merged.stdout)
            if merged.returncode != 0 or not tree:
                return pr.number
            result = _git(
                self.repo_root,
                "commit-tree", tree, "-p", accumulated, "-p", head,
                "-m", f"probe: replay #{pr.number}",
                env=PROBE_IDENTITY,
            )
            accumulated = result.stdout.strip()
            if not accumulated:
                detail = result.stderr.strip() or f"exit {result.returncode}"
                raise RuntimeError(
                    f"failed to synthesize replay commit for #{pr.number}: {detail}"
                )
        return None

    def is_ancestor(self, ancestor: PullRequest, descendant: PullRequest) -> bool:
        """True when ``ancestor``'s head commit is reachable from ``descendant``'s.

        This is the ground truth ``baseRefName`` only claims. Unresolvable refs
        answer False: an unknown relationship must not be reported as a
        confirmed stack, because that would skip the collision probe.
        """
        a_sha = rev_parse_commit(self.repo_root, ancestor.head_ref)
        b_sha = rev_parse_commit(self.repo_root, descendant.head_ref)
        if not a_sha or not b_sha or a_sha == b_sha:
            return False
        return _git(self.repo_root, "merge-base", "--is-ancestor", a_sha, b_sha).returncode == 0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_report(report: MergeOrderReport, prs: Iterable[PullRequest]) -> str:
    title_of = {pr.number: pr.title for pr in prs}
    lines: list[str] = []

    lines.append(
        f"Probed {len(report.prs)} open PRs: " + " ".join(f"#{n}" for n in report.prs)
    )
    lines.append("")

    if report.stacked_on:
        lines.append("Stacked (hard ordering edge, not a conflict):")
        for child, parent in sorted(report.stacked_on.items()):
            lines.append(f"  #{child} sits on top of #{parent}")
        lines.append("")

    if report.diverged:
        lines.append("DIVERGED STACK — declared base is not an ancestor, probed as siblings:")
        for child, parent in report.diverged:
            lines.append(f"  #{child} declares #{parent} but does not descend from it")
        lines.append("")

    if report.mutual_conflicts:
        # Conflicts in both directions prove only that these branches cannot
        # both land unchanged. They say nothing about which PR is right, or
        # that either is redundant — telling an operator to close one is a
        # destructive instruction the evidence does not support.
        lines.append(
            "MUTUALLY CONFLICTING — decide which lands first, then rebase or resolve the other:"
        )
        for pair in report.mutual_conflicts:
            lines.append(f"  #{pair.a} <-> #{pair.b}")
            for file in pair.files:
                lines.append(f"      {file}")
        lines.append("")

    ordered = [p for p in report.pairs if p.relation == "ordered"]
    if ordered:
        lines.append("Ordered (one direction merges clean):")
        for pair in ordered:
            second = pair.b if pair.first == pair.a else pair.a
            lines.append(
                f"  #{pair.first} must land before #{second} ({', '.join(pair.files)})"
            )
        lines.append("")

    lines.append("Viable merge order:")
    if not report.merge_order:
        lines.append("  (none — resolve the conflicts above first)")
    else:
        for index, n in enumerate(report.merge_order, start=1):
            lines.append(f"  {index}. #{n} {title_of.get(n, '')}".rstrip())

    unverified = set(report.unverified)
    held = [n for n in report.blocked if n not in report.merge_order and n not in unverified]
    if held:
        lines.append("")
        lines.append("Held back pending a decision: " + " ".join(f"#{n}" for n in held))
    if report.unverified:
        lines.append("")
        lines.append(
            "No verdict (held out of the order): "
            + " ".join(f"#{n}" for n in report.unverified)
        )

    # A queue can hold more than one trunk. Naming the split is the honest
    # report: silently probing a release-branch PR against main invents verdicts.
    by_base: dict[str, list[int]] = {}
    for n in report.prs:
        by_base.setdefault(report.bases.get(n, ""), []).append(n)
    if len(by_base) > 1:
        lines.append("")
        lines.append("Separate queues — PRs on different trunks are never probed against each other:")
        for base, members in sorted(by_base.items(), key=lambda kv: -len(kv[1])):
            lines.append(
                f"  {base or '(unknown base)'}: " + " ".join(f"#{n}" for n in members)
            )

    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for error in report.errors:
            lines.append(f"  {error}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# queue discovery
# ---------------------------------------------------------------------------


SLUG_RE = re.compile(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?$")


def resolve_repo_slug(repo_root: Path, remote: str = "origin") -> str:
    """Derives ``owner/name`` from a remote URL rather than letting ``gh`` guess.

    This checkout carries three remotes (``origin`` and ``pai-scaffolde`` for
    the fork, ``upstream`` for NousResearch). With no default set, ``gh pr
    list`` resolves to the *upstream* repository and returns its ~85,000-PR
    queue: a sweep that trusted that resolution would probe a queue this
    repository never lands, and report an order for PRs that do not exist here.
    Pinning the slug to a named remote removes the guess.
    """
    result = _git(repo_root, "remote", "get-url", remote)
    if result.returncode != 0:
        raise RuntimeError(f"cannot read remote {remote}: {result.stderr.strip()}")
    match = SLUG_RE.search(result.stdout.strip())
    if not match:
        raise RuntimeError(f"cannot parse owner/name from remote {remote}: {result.stdout.strip()}")
    return match.group(1)


def gh_rest_open_pr_count(repo_root: Path, slug: str) -> int:
    """REST-side open-PR count. Deliberately a different API surface to ``gh pr list``."""
    result = subprocess.run(
        [
            "gh", "api", "-X", "GET", f"repos/{slug}/pulls",
            "-f", "state=open", "-f", "per_page=100", "--jq", "length",
        ],
        cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"corroborating gh api pulls failed: {result.stderr.strip()}")
    raw = result.stdout.strip()
    # ``int("")`` raises but ``bool("0")`` is True and a bare falsy coercion
    # would read a corroborator that answered NOTHING as a corroborated zero —
    # the exact fail-open this second opinion exists to close. ``--jq length``
    # emits a bare non-negative integer or nothing; demand the digits.
    if not re.fullmatch(r"\d+", raw):
        raise RuntimeError(f"corroborating gh api pulls returned no count: {raw!r}")
    return int(raw)


def assert_queue_genuinely_empty(corroborate: Callable[[], int]) -> None:
    """An empty queue must be corroborated before it is believed.

    ``gh pr list`` exiting 0 with ``[]`` is indistinguishable from a genuinely
    empty queue, and every downstream consumer reads it as "verified, nothing
    to do". That is a fail-open: a query that answers nothing — wrong auth,
    wrong repo resolution, a truncated response — reports the same clean sweep
    as a queue that really is empty.

    No single query can tell those apart, so the second opinion comes from a
    DIFFERENT code path: the primary count from ``gh pr list`` (GraphQL), the
    corroborating one from the REST pulls endpoint. Disagreement raises rather
    than warns — a sweep that cannot establish what it was meant to examine has
    not verified the queue.
    """
    second = corroborate()
    if second > 0:
        raise RuntimeError(
            f"queue query returned no open PRs, but an independent REST count found {second}. "
            "The PR query is not answering — refusing to report a swept queue."
        )


def fetch_open_pull_requests(
    repo_root: Path,
    slug: str,
    limit: int,
    corroborate: Callable[[], int] | None = None,
) -> list[PullRequest]:
    result = subprocess.run(
        [
            "gh", "pr", "list", "--repo", slug, "--state", "open",
            "--limit", str(limit),
            "--json", "number,headRefName,baseRefName,title,isDraft,isCrossRepository",
        ],
        cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh pr list failed: {result.stderr.strip()}")
    parsed = json.loads(result.stdout)
    # ``--limit`` is a MAXIMUM, and gh truncates silently. A queue larger than
    # the cap would be swept as if the omitted PRs did not exist, so a collision
    # involving one of them reads as a clean, landable verdict. A sweep that saw
    # part of the queue has not verified the queue: refuse rather than answer.
    if len(parsed) >= limit:
        raise RuntimeError(
            f"gh pr list returned {len(parsed)} PRs at --limit {limit}: the queue is "
            "truncated and was NOT fully examined. Re-run with a higher --limit."
        )
    # Corroborate the QUERY, not the filtered result. A queue of nothing but
    # drafts legitimately yields zero probe targets from a query that answered
    # correctly; demanding a second opinion there would fail on a healthy repo.
    # Only an empty *response* is the ambiguous case.
    if not parsed:
        assert_queue_genuinely_empty(
            corroborate or (lambda: gh_rest_open_pr_count(repo_root, slug))
        )
    return [
        PullRequest(
            number=pr["number"],
            head_ref=pr["headRefName"],
            base_ref=pr["baseRefName"],
            title=pr.get("title", ""),
            cross_repository=bool(pr.get("isCrossRepository")),
        )
        for pr in parsed
        if not pr.get("isDraft")
    ]


def remote_tracking_ref(remote: str, base_ref: str) -> str:
    return f"refs/remotes/{remote}/{base_ref}"


def build_fetch_args(prs: list[PullRequest], remote: str) -> list[str]:
    """Every head AND every base has to be present locally, and current.

    Fetching only the PR refspecs left the base behind: ``git fetch <remote>
    <refspec>...`` updates exactly the refs named, so ``origin/main`` keeps
    whatever SHA it had from the last unrelated fetch, and the sweep probes
    every pair against an obsolete trunk. ``--no-write-fetch-head`` keeps the
    advertised read-only behaviour literal — the default invocation *does*
    write FETCH_HEAD, clobbering a ref another workflow may be mid-read on.
    """
    refspecs = [
        f"+refs/pull/{pr.number}/head:{PROBE_REF_NAMESPACE}{pr.number}" for pr in prs
    ]
    for base in sorted({pr.base_ref for pr in prs if pr.base_ref}):
        refspecs.append(f"+refs/heads/{base}:{remote_tracking_ref(remote, base)}")
    return ["fetch", "--quiet", "--no-write-fetch-head", remote, *refspecs]


def fetch_heads(repo_root: Path, prs: list[PullRequest], remote: str) -> list[str]:
    if not prs:
        return []
    result = _git(repo_root, *build_fetch_args(prs, remote))
    if result.returncode != 0:
        return [f"fetch failed: {result.stderr.strip()}"]
    return []


def prune_probe_refs(repo_root: Path, keep: Iterable[int]) -> list[str]:
    """Drops ``refs/pr-merge-order/<n>`` for every PR no longer in the queue.

    A closed PR's ref is never named by a later fetch, so it was never updated
    or deleted: the namespace only grew, each entry keeping an obsolete commit
    reachable and therefore un-collectable by ``git gc``. A tool that must stay
    safe to run during a disk-pressure outage cannot leak disk every run.
    """
    keep_set = {str(n) for n in keep}
    listed = _git(repo_root, "for-each-ref", "--format=%(refname)", PROBE_REF_NAMESPACE)
    if listed.returncode != 0:
        return []
    pruned: list[str] = []
    for ref in (line.strip() for line in listed.stdout.split("\n")):
        if not ref:
            continue
        if ref[len(PROBE_REF_NAMESPACE):] in keep_set:
            continue
        if _git(repo_root, "update-ref", "-d", ref).returncode == 0:
            pruned.append(ref)
    return pruned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument("--repo", help="owner/name to sweep (default: derived from --remote)")
    parser.add_argument("--remote", default="origin", help="git remote to fetch and resolve from")
    parser.add_argument("--limit", type=int, default=100, help="max open PRs to consider")
    args = parser.parse_args(argv)

    repo_root = REPO_ROOT
    slug = args.repo or resolve_repo_slug(repo_root, args.remote)
    # Name what is being probed. Silent repo resolution is how a sweep ends up
    # reporting on the upstream queue instead of this one.
    print(f"pr-merge-order: probing {slug} via remote {args.remote} in {repo_root}", file=sys.stderr)

    prs = fetch_open_pull_requests(repo_root, slug, args.limit)
    if not prs:
        # Prune before the early return. Skipping it here leaves every ref an
        # earlier run created under refs/pr-merge-order/, and in a persistent
        # checkout those refs keep closed PR histories reachable forever —
        # unbounded disk growth from the cleanup path that was meant to bound it.
        prune_probe_refs(repo_root, [])
        empty = MergeOrderReport()
        print(json.dumps(empty.to_dict(), indent=2) if args.json else "No open PRs.")
        return EXIT_LANDABLE

    fetch_errors = fetch_heads(repo_root, prs, args.remote)
    prune_probe_refs(repo_root, [pr.number for pr in prs])

    # Probe the namespaced local refs, not the remote branch names: a PR from a
    # fork has no local branch, and a stale origin/<branch> would probe the
    # wrong commit. Stacking is decided from the real branch names, so the base
    # link is re-expressed in the same namespace.
    by_number = {pr.number: pr for pr in prs}
    declared_parent = declared_parents(prs)

    probe_targets = [
        PullRequest(
            number=pr.number,
            head_ref=f"{PROBE_REF_NAMESPACE}{pr.number}",
            base_ref=(
                f"{PROBE_REF_NAMESPACE}{declared_parent[pr.number]}"
                if pr.number in declared_parent
                else remote_tracking_ref(args.remote, pr.base_ref)
            ),
            title=pr.title,
            cross_repository=pr.cross_repository,
        )
        for pr in prs
    ]

    # Each PR is probed against ITS trunk — the base of the root of its stack. A
    # stacked head already carries its ancestors' commits, so merging it onto
    # the trunk is the right three-way merge.
    def trunk_of(pr: PullRequest) -> str:
        cursor, seen = pr.number, {pr.number}
        while cursor in declared_parent and declared_parent[cursor] not in seen:
            cursor = declared_parent[cursor]
            seen.add(cursor)
        return remote_tracking_ref(args.remote, by_number[cursor].base_ref)

    suite = GitProbeSuite(repo_root, trunk_of)
    report = compute_merge_order(
        probe_targets,
        suite.probe,
        suite.is_ancestor,
        preflight=suite.preflight,
        replay=suite.replay,
    )
    report.errors.extend(fetch_errors)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, prs))

    return exit_code_for(report)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - top-level CLI boundary
        print(f"pr-merge-order failed: {error}", file=sys.stderr)
        sys.exit(2)
