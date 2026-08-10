"""This process's own cgroup memory ceiling.

``/sys/fs/cgroup/memory.max`` is the *hierarchy root*.  It is the right
file to read only for a process in the root cgroup.  Under a systemd unit
with ``MemoryMax=``, or in a container without a private cgroup namespace,
the root reads ``max`` while the limit that actually binds the process
sits under the path named in ``/proc/self/cgroup``.  Reading the root
there reports "unlimited" and hands the caller the node's RAM.

Two callers size a budget from that answer and are wrong in the same way
when it is the root's:

- ``agent.lsp.manager`` caps the concurrent language-server fleet.
- ``hermes_cli.main`` sizes the Node TUI's V8 ``--max-old-space-size``.

Stdlib only, and deliberately free of any LSP or CLI import: both callers
reach it on a startup path.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


# Hierarchy roots.  These are only correct for a process in the root
# cgroup; they are the fallback for hosts with no ``/proc`` to consult.
CGROUP_MEMORY_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",  # v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
)

# Prefixed onto the absolute paths the ``/proc`` resolution reads.  Empty
# is the real filesystem; tests point it at a fabricated ``/proc`` plus
# cgroupfs so path resolution is exercised rather than stubbed out.
CGROUP_FS_ROOT = ""

# "``/proc`` could not answer", which is not the same as "``/proc`` says
# unlimited".  Only the former may fall through to the hierarchy roots.
_UNRESOLVED = object()

# v2 unifies every controller under one hierarchy; v1 gives the memory
# controller its own mount, with its own filename.
_CGROUP_LIMIT_FILES = (("v2", "memory.max"), ("memory", "memory.limit_in_bytes"))


def _rooted(path: str) -> str:
    return CGROUP_FS_ROOT + path if CGROUP_FS_ROOT else path


def _parse_memory_limit(raw: str) -> Optional[int]:
    """Bytes from a limit file's text, or ``None`` for unlimited/garbage.

    cgroup v2 writes ``max`` for "no limit"; v1 writes a sentinel near
    2**63, so implausibly large values are treated as absent.
    """
    raw = raw.strip()
    if not raw or raw == "max":
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    # v1's "unlimited" sentinel is page-aligned LONG_MAX; anything at
    # petabyte scale is that sentinel rather than a real allocation.
    if limit <= 0 or limit >= (1 << 50):
        return None
    return limit


def _read_memory_limit(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _parse_memory_limit(fh.read())
    except (OSError, ValueError):
        return None


def _unescape_mount_field(field: str) -> str:
    """mountinfo octal-escapes the bytes that would break its own parse."""
    for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        field = field.replace(code, char)
    return field


def _cgroup_mounts() -> Dict[str, List[Tuple[str, str]]]:
    """``{kind: [(mount_root, mount_point), ...]}`` for the memory hierarchies.

    A mountinfo line is ``id parent major:minor root mount_point opts
    [optional...] - fstype source super_opts``.  The optional fields are
    variable in number, which is why the ``-`` separator splits the line
    instead of a fixed index.
    """
    mounts: Dict[str, List[Tuple[str, str]]] = {}
    try:
        with open(_rooted("/proc/self/mountinfo"), "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except (OSError, ValueError):
        return mounts
    for line in lines:
        head, sep, tail = line.partition(" - ")
        if not sep:
            continue
        head_fields = head.split()
        tail_fields = tail.split()
        if len(head_fields) < 5 or len(tail_fields) < 3:
            continue
        mount = (_unescape_mount_field(head_fields[3]), _unescape_mount_field(head_fields[4]))
        fstype, super_opts = tail_fields[0], tail_fields[2]
        # Every match is kept.  A namespace can expose several cgroup2 or
        # v1-memory mounts, and the first listed may be a bind mount rooted
        # outside this process's cgroup — which can say nothing about its
        # limit.  The caller picks among them; order here is mountinfo's.
        if fstype == "cgroup2":
            mounts.setdefault("v2", []).append(mount)
        elif fstype == "cgroup" and "memory" in super_opts.split(","):
            mounts.setdefault("memory", []).append(mount)
    return mounts


def _proc_self_cgroups() -> Dict[str, str]:
    """``{kind: cgroup_path}`` for this process, from ``/proc/self/cgroup``.

    v2 writes ``0::/<path>``; v1 writes ``<hid>:<controllers>:/<path>``.
    """
    found: Dict[str, str] = {}
    try:
        with open(_rooted("/proc/self/cgroup"), "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except (OSError, ValueError):
        return found
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hid, controllers, path = parts
        if not path.startswith("/"):
            continue
        if hid == "0" and not controllers:
            found.setdefault("v2", path)
        elif "memory" in controllers.split(","):
            found.setdefault("memory", path)
    return found


def _cgroup_dir(cgroup_path: str, mount_root: str, mount_point: str) -> Optional[str]:
    """Where *cgroup_path* lives under a mount of *mount_root*, or ``None``.

    A mount can expose a subtree: with ``/system.slice`` mounted at
    ``/sys/fs/cgroup``, this process's ``/system.slice/hermes.service``
    is on disk at ``/sys/fs/cgroup/hermes.service``.

    ``None`` means this mount cannot see the path at all — a bind mount
    of some unrelated subtree.  Its root is not a weaker answer about
    our limit, it is an answer about a different cgroup, so the caller
    must prefer a mount that maps rather than read this one.
    """
    trimmed = mount_root.rstrip("/")
    rel = cgroup_path
    if trimmed:
        if cgroup_path == trimmed:
            rel = "/"
        elif cgroup_path.startswith(trimmed + "/"):
            rel = cgroup_path[len(trimmed):]
        else:
            return None
    base = _rooted(mount_point)
    if rel in ("", "/"):
        return base
    return os.path.join(base, rel.lstrip("/"))


def _limit_up_to_mount(start: str, ceiling: str, filename: str) -> Optional[int]:
    """Smallest finite limit from *start* up to *ceiling*, inclusive.

    An ancestor's limit binds its descendants, so a slice capped below
    its child's own cap is the one that applies; the walk takes the
    minimum rather than stopping at the first limit it finds.
    """
    top = os.path.normpath(ceiling)
    current = os.path.normpath(start)
    best: Optional[int] = None
    while True:
        limit = _read_memory_limit(os.path.join(current, filename))
        if limit is not None:
            best = limit if best is None else min(best, limit)
        if current == top:
            break
        parent = os.path.dirname(current)
        if parent == current or len(parent) < len(top):
            break
        current = parent
    return best


def _proc_cgroup_memory_limit_bytes() -> Any:
    """This process's own cgroup ceiling, or ``_UNRESOLVED``.

    ``/sys/fs/cgroup/memory.max`` is the *hierarchy root*.  Under a
    systemd unit with ``MemoryMax=``, or in a container without a private
    cgroup namespace, the root reads ``max`` while the limit that binds
    the process sits under the path named in ``/proc/self/cgroup``.
    """
    cgroups = _proc_self_cgroups()
    if not cgroups:
        return _UNRESOLVED
    mounts = _cgroup_mounts()
    resolved = False
    best: Optional[int] = None
    # Both hierarchies are consulted rather than the first that matches: a
    # hybrid host mounts cgroup2 without the memory controller, which lives
    # on v1.
    for kind, filename in _CGROUP_LIMIT_FILES:
        cgroup_path = cgroups.get(kind)
        if cgroup_path is None:
            continue
        candidates: List[Tuple[str, str]] = []
        for mount_root, mount_point in mounts.get(kind, ()):
            start = _cgroup_dir(cgroup_path, mount_root, mount_point)
            if start is not None:
                candidates.append((start, _rooted(mount_point)))
        # Only when nothing maps this process's path is a mount root worth
        # reading, and then it is the pre-existing fixed-path answer.
        if not candidates:
            candidates = [
                (_rooted(mount_point), _rooted(mount_point))
                for _, mount_point in mounts.get(kind, ())
            ]
        for start, ceiling in candidates:
            resolved = True
            limit = _limit_up_to_mount(start, ceiling, filename)
            if limit is not None:
                best = limit if best is None else min(best, limit)
    return best if resolved else _UNRESOLVED


def _cgroup_memory_limit_bytes(paths: Optional[Tuple[str, ...]] = None) -> Optional[int]:
    """The cgroup memory ceiling in bytes, or ``None`` if unlimited/absent.

    Checked before trusting sysconf because in a memory-limited container
    ``SC_PHYS_PAGES`` reports the *node's* RAM, not the share this process
    may actually use.

    Resolution goes through ``/proc/self/cgroup`` so the answer is the
    limit on *this* process rather than on the hierarchy root.  When
    ``/proc`` cannot answer — macOS, Windows, a sandbox that hides it —
    the fixed hierarchy-root paths are the fallback.

    *paths* is read from the module attribute by default; passing it
    explicitly selects the fallback alone, keeping the sentinel handling
    testable in isolation.
    """
    if paths is None:
        resolved = _proc_cgroup_memory_limit_bytes()
        if resolved is not _UNRESOLVED:
            return resolved
    for path in paths if paths is not None else CGROUP_MEMORY_LIMIT_PATHS:
        limit = _read_memory_limit(path)
        if limit is not None:
            return limit
    return None


# The public name across module boundaries; the resolution helpers above
# stay private to this module.
cgroup_memory_limit_bytes = _cgroup_memory_limit_bytes
