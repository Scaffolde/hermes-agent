#!/usr/bin/env python3
"""Measure the real memory footprint of each language server, per ``server_id``.

Why this exists
---------------

``agent.lsp.manager`` bounds the language-server fleet against a memory budget.
Sizing that budget needs a per-type cost, and the cost cannot be guessed: a
language server's memory is dominated by the program it loads, and the spread
across types is more than an order of magnitude (a Node shim parsing shell
scripts versus tsserver holding a typed project graph).

It also cannot be read from ``ps``.  Every Node-based server is launched through
a shim whose RSS reads 5-11 MB while the actual analysis lives in a child
process, and on macOS RSS excludes compressed and swapped pages — which is
precisely the memory that caused SCA-4389.  ``vmmap --summary``'s *physical
footprint* is the number that tracks what the kernel charges the process, so
that is what this measures, summed over the whole spawned subtree.

Usage
-----

    python scripts/measure_lsp_footprint.py --server typescript --server pyright
    python scripts/measure_lsp_footprint.py --all --json out.json

Each measurement spawns a real server against a real workspace, opens a real
file, waits for diagnostics to settle, and then measures.  Servers that are not
installed are reported as ``skipped`` rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# How long to let a server index before measuring.  A server measured too early
# reports its startup footprint rather than its working one, which would bias
# every value downward — the direction that re-opens SCA-4389.
SETTLE_SECONDS = 20.0

# Sampling: a single reading can land in a GC trough.  Take the peak across a
# few samples, because the cap has to hold against the peak, not the average.
SAMPLE_COUNT = 3
SAMPLE_INTERVAL = 4.0


# ---------------------------------------------------------------------------
# process-tree footprint
# ---------------------------------------------------------------------------


def _child_pids(root_pid: int) -> List[int]:
    """Every pid in the subtree rooted at *root_pid*, including itself.

    Walks ``ps -eo pid,ppid`` rather than asking for descendants directly
    because the shim->real-server relationship is exactly what we must not
    miss: measuring only *root_pid* is how you get the 5-11 MB reading that
    makes a 1.4 GiB server look free.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return [root_pid]

    children: Dict[int, List[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)

    seen: List[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children.get(pid, ()))
    return seen


def _physical_footprint_macos(pid: int) -> Optional[int]:
    """Physical footprint of *pid* in bytes via ``vmmap --summary``.

    Returns ``None`` when the process is gone or vmmap is not permitted,
    so a partial tree is visibly partial rather than silently undercounted.
    """
    try:
        proc = subprocess.run(
            ["vmmap", "--summary", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if "Physical footprint:" in line and "peak" not in line.lower():
            value = line.split(":", 1)[1].strip()
            return _parse_size(value)
    return None


def _physical_footprint_linux(pid: int) -> Optional[int]:
    """Linux equivalent: anonymous RSS + swap from ``/proc/<pid>/status``.

    Swap is included for the same reason macOS uses physical footprint —
    paged-out anonymous memory is still memory this fleet caused to be
    allocated, and excluding it is what makes an over-subscribed host look
    healthy right up until it thrashes.
    """
    total = 0
    found = False
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("RssAnon:", "VmSwap:")):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            total += int(parts[1]) * 1024
                            found = True
                        except ValueError:
                            pass
    except OSError:
        return None
    return total if found else None


def _parse_size(value: str) -> Optional[int]:
    """Parse vmmap's ``1.4G`` / ``938.2M`` / ``12K`` into bytes."""
    value = value.strip()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    multiplier = 1
    if value and value[-1].upper() in units:
        multiplier = units[value[-1].upper()]
        value = value[:-1]
    try:
        return int(float(value) * multiplier)
    except ValueError:
        return None


def tree_footprint_bytes(root_pid: int) -> Tuple[Optional[int], int]:
    """(summed footprint, pids measured) for the subtree at *root_pid*."""
    reader = (
        _physical_footprint_macos
        if platform.system() == "Darwin"
        else _physical_footprint_linux
    )
    total = 0
    counted = 0
    for pid in _child_pids(root_pid):
        value = reader(pid)
        if value is not None:
            total += value
            counted += 1
    if counted == 0:
        return None, 0
    return total, counted


# ---------------------------------------------------------------------------
# per-server measurement
# ---------------------------------------------------------------------------

def _find_sample_file(server_id: str, workspace: str) -> Optional[str]:
    """A real file of the right type inside *workspace*, or ``None``."""
    from agent.lsp.servers import SERVERS

    server = next((s for s in SERVERS if s.server_id == server_id), None)
    if server is None:
        return None
    extensions = tuple(server.extensions)
    best: Optional[Tuple[int, str]] = None
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in {"node_modules", "__pycache__"}
        ]
        for name in filenames:
            if not name.endswith(extensions):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            # Prefer a substantial file: a one-line file may not pull in
            # enough of the project to represent real cost.
            if best is None or size > best[0]:
                best = (size, path)
        if best is not None and best[0] > 40_000:
            break
    return best[1] if best else None


def measure_server(server_id: str, workspace: str, *, verbose: bool = True) -> Dict:
    """Spawn one server, let it settle, and measure its subtree."""
    from agent.lsp.manager import LSPService

    result: Dict = {"server_id": server_id, "workspace": workspace}

    sample = _find_sample_file(server_id, workspace)
    if sample is None:
        result["status"] = "skipped"
        result["reason"] = "no sample file of this type in the workspace"
        return result
    result["sample_file"] = os.path.relpath(sample, workspace)

    # idle_timeout=0 disables the reaper, and a cap of 24 keeps this
    # measurement's own server from being evicted mid-reading.
    manager = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=30.0,
        # A measurement run must be able to obtain the server it is measuring;
        # refusing to install here would report "skipped" for every type this
        # host has not happened to install by hand.
        install_strategy="auto",
        idle_timeout=0,
        max_clients=24,
    )
    try:
        if verbose:
            print(f"  spawning {server_id} against {result['sample_file']} ...", flush=True)
        try:
            manager.get_diagnostics_sync(sample, delta=False, timeout=120.0)
        except Exception as e:  # noqa: BLE001
            result["status"] = "error"
            result["reason"] = f"diagnostics failed: {e}"
            return result

        client = next(
            (c for (sid, _root), c in manager._clients.items() if sid == server_id),
            None,
        )
        if client is None or client._proc is None:
            result["status"] = "skipped"
            result["reason"] = "server did not start (not installed?)"
            return result

        pid = client._proc.pid
        result["pid"] = pid

        if verbose:
            print(f"  settling {SETTLE_SECONDS}s ...", flush=True)
        time.sleep(SETTLE_SECONDS)

        peak = 0
        samples: List[int] = []
        for i in range(SAMPLE_COUNT):
            if i:
                time.sleep(SAMPLE_INTERVAL)
            total, counted = tree_footprint_bytes(pid)
            if total is None:
                continue
            samples.append(total)
            if total > peak:
                peak = total
                result["pids_measured"] = counted

        if not samples:
            result["status"] = "error"
            result["reason"] = "no footprint reading (vmmap denied or process gone)"
            return result

        result["status"] = "measured"
        result["footprint_bytes"] = peak
        result["footprint_mib"] = round(peak / (1024**2), 1)
        result["samples_mib"] = [round(s / (1024**2), 1) for s in samples]
        return result
    finally:
        try:
            manager.shutdown()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        action="append",
        default=[],
        help="server_id to measure (repeatable)",
    )
    parser.add_argument("--all", action="store_true", help="measure every registered server")
    parser.add_argument(
        "--workspace",
        default=REPO_ROOT,
        help="project to measure against (default: this repo)",
    )
    parser.add_argument("--json", help="write results to this path")
    args = parser.parse_args()

    from agent.lsp.servers import SERVERS

    if args.all:
        targets = [s.server_id for s in SERVERS]
    elif args.server:
        targets = args.server
    else:
        parser.error("pass --server <id> (repeatable) or --all")
        return 2

    host: Dict = {
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    if shutil.which("sysctl"):
        try:
            host["memsize_bytes"] = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    print(f"host: {host}")
    print(f"workspace: {args.workspace}")
    results = []
    for server_id in targets:
        print(f"\n== {server_id}")
        res = measure_server(server_id, args.workspace)
        status = res.get("status")
        if status == "measured":
            print(f"  -> {res['footprint_mib']} MiB across {res.get('pids_measured')} pids")
        else:
            print(f"  -> {status}: {res.get('reason')}")
        results.append(res)

    payload = {"host": host, "workspace": args.workspace, "results": results}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    else:
        print("\n" + json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
