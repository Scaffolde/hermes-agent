#!/usr/bin/env bash
# Decide the job outcome for a pr_merge_order.py run (SCA-4638).
#
# This lives outside the workflow YAML on purpose. While the decision was
# inline shell, the only way to test it was to grep the workflow text for
# substrings — which passes whenever the string is present regardless of
# whether the logic is wired correctly, and fails on a behaviour-preserving
# reformat. Here it is a real boundary: give it a status, observe an exit code
# and a verdict line.
#
# Usage: pr_merge_order_gate.sh <status>
#   stdout — the one-line verdict for the job summary
#   exit 0 — the sweep produced a verdict (queue landable, or a real collision
#            the caller should now report to a human)
#   exit 1 — the queue was NOT verified; the job must go red
#
# Status 1 exits 0 here because a detected collision is a successful detection.
# Failing the job on it would make a working detector indistinguishable from a
# broken one, which is the confusion this whole tool exists to remove.

set -uo pipefail

status="${1-}"

if [ -z "$status" ]; then
  echo "pr-merge-order gate: no status supplied" >&2
  exit 1
fi

case "$status" in
  0)
    echo "The queue has a landable order."
    exit 0
    ;;
  1)
    echo "**The queue cannot land as it stands.** See the collisions below."
    exit 0
    ;;
  2)
    echo "**The queue was NOT verified** (status 2). The sweep did not establish a verdict for every PR; this is not a clean result."
    echo "::error::pr-merge-order could not verify the queue (status 2)" >&2
    exit 1
    ;;
  *)
    # 127 (no interpreter), 137 (killed), or a traceback's own status. Checking
    # only for 2 let these through: 127 read as clean, and a script that failed
    # to parse exited 1 and was reported as a real queue collision.
    echo "**The queue was NOT verified** (status $status, outside the documented 0/1/2 contract)."
    echo "::error::pr-merge-order exited $status, outside its 0/1/2 contract; the queue was NOT verified" >&2
    exit 1
    ;;
esac
