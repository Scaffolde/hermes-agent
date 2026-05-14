#!/usr/bin/env python3
"""Verify Scaffolde fork patch markers are present in the current tree."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _write_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default=os.environ.get("PATCHES_MANIFEST", ".scaffolde-patches.yaml"),
        help="Path to the Scaffolde patch marker manifest",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:
        report = [
            "## Hermes upstream sync — patches manifest missing or invalid",
            "",
            f"`{manifest_path}` could not be read: {exc}",
            "Refusing to push the sync branch.",
        ]
        _write_summary(report)
        print("\n".join(report), file=sys.stderr)
        return 1

    entries = manifest.get("patches", []) or []
    failures: list[str] = []
    checked_markers = 0
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(f"manifest entry is not a mapping: {entry!r}")
            continue
        path = entry.get("path")
        needles = entry.get("contains") or []
        if not path:
            failures.append(f"manifest entry missing 'path': {entry!r}")
            continue
        patch_path = Path(path)
        if not patch_path.exists():
            failures.append(f"{path}: file missing after merge")
            continue
        content = patch_path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            checked_markers += 1
            if needle not in content:
                failures.append(f"{path}: missing marker {needle!r}")

    if failures:
        report = ["## Hermes upstream sync — patch survival check FAILED", ""]
        report.append(f"{len(failures)} marker(s) missing after merging upstream:")
        report.append("")
        for failure in failures:
            report.append(f"- {failure}")
        report.extend(
            [
                "",
                "Refusing to push the sync branch.",
                "If a marker was intentionally renamed, update `.scaffolde-patches.yaml` in the same PR.",
            ]
        )
        _write_summary(report)
        print("\n".join(report), file=sys.stderr)
        return 1

    print(
        f"All {checked_markers} marker(s) across {len(entries)} file(s) in "
        f"{manifest_path} survived the merge."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
