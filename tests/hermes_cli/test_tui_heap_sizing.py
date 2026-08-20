"""Tests for cgroup-aware TUI V8 heap sizing.

V8 is not cgroup-aware: a flat ``--max-old-space-size=8192`` lets the heap grow
toward 8GB in a memory-limited container, so the cgroup OOM-killer SIGKILLs Node
before V8's own monitor fires — leaving the user with only a bare gateway
``stdin EOF`` and no breadcrumb. ``_resolve_tui_heap_mb`` reads the real cgroup
limit and sizes the cap below it so V8 exits gracefully instead.
"""

import builtins
import io
from unittest import mock

import hermes_cli.main as m
from agent import cgroup_memory as cgroup_mod

V2 = "/sys/fs/cgroup/memory.max"
V1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
GB = 1024 ** 3


def _fake_open(files: dict):
    """Return an open() shim serving cgroup paths from ``files`` (path->str)."""
    real_open = builtins.open

    def opener(path, *args, **kwargs):
        if path in (V2, V1):
            content = files.get(path)
            if content is None:
                raise FileNotFoundError(path)
            return io.StringIO(content)
        return real_open(path, *args, **kwargs)

    return opener


def _no_proc_root(tmp_path) -> str:
    """A root with no ``/proc``, so resolution falls through to the fixed paths.

    ``_fake_open`` passes every path it does not recognise to the real
    ``open``.  Once resolution consults ``/proc/self/cgroup``, that would read
    the *real* ``/proc`` on Linux CI and the fixed-path tests would stop being
    deterministic — they would report the CI runner's own cgroup.  Pinning the
    root at an empty directory is the same guard ``no_proc()`` gives the LSP
    suite in ``tests/agent/lsp/test_client_cap.py``.
    """
    empty = tmp_path / "no-proc-root"
    empty.mkdir(exist_ok=True)
    return str(empty)


def _read(files: dict, tmp_path):
    with mock.patch.object(cgroup_mod, "CGROUP_FS_ROOT", _no_proc_root(tmp_path)):
        with mock.patch.object(builtins, "open", _fake_open(files)):
            return m._read_cgroup_memory_limit()


# ----------------------------------------------------------------------
# cgroup path resolution
#
# The fixed-path tests above prove the parsing, and can say nothing about
# *which file* a deployed process reads.  ``/sys/fs/cgroup/memory.max`` is
# the hierarchy root: under a systemd unit with ``MemoryMax=``, or in a
# container without a private cgroup namespace, it reads ``max`` while the
# limit that binds the process lives at the path named in
# ``/proc/self/cgroup``.  These build a whole fake ``/proc`` + cgroupfs so
# the path resolution itself is the thing under test.
# ----------------------------------------------------------------------

V2_MOUNTINFO = (
    "31 24 0:27 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime"
    " shared:9 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n"
)


def fake_root(tmp_path, *, cgroup: str, mountinfo: str, limits: dict) -> str:
    """Materialise a fake filesystem root and return its path."""
    root = tmp_path / "root"
    proc = root / "proc" / "self"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "cgroup").write_text(cgroup, encoding="utf-8")
    (proc / "mountinfo").write_text(mountinfo, encoding="utf-8")
    for abs_path, text in limits.items():
        target = root / abs_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return str(root)


def _constrained_unit(tmp_path, limit_bytes: int) -> str:
    """A 4 GiB-ish systemd unit whose hierarchy root reads ``max``."""
    return fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=V2_MOUNTINFO,
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{limit_bytes}\n",
        },
    )


class TestLimitComesFromThisProcessCgroup:
    def test_unit_limit_is_read_not_the_hierarchy_root(self, monkeypatch, tmp_path):
        """The root reads ``max``; the unit's own cgroup carries the real cap."""
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", _constrained_unit(tmp_path, 4 * GB))

        assert m._read_cgroup_memory_limit() == 4 * GB

    def test_heap_is_sized_below_the_unit_limit_not_the_flat_default(self, monkeypatch, tmp_path):
        """The consequence: a flat 8192 grows the heap into the OOM killer.

        V8 is not cgroup-aware, so an 8 GB cap inside a 4 GiB unit is a
        SIGKILL with no JS handler, no ``[tui-parent]`` breadcrumb, and a
        bare gateway ``stdin EOF`` for the user.
        """
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", _constrained_unit(tmp_path, 4 * GB))

        assert m._resolve_tui_heap_mb() == 3072

    def test_an_ancestor_limit_binds_the_unit(self, monkeypatch, tmp_path):
        """A slice capped below its child's own cap is the one that applies."""
        root = fake_root(
            tmp_path,
            cgroup="0::/system.slice/hermes-agent.service\n",
            mountinfo=V2_MOUNTINFO,
            limits={
                "/sys/fs/cgroup/memory.max": "max\n",
                "/sys/fs/cgroup/system.slice/memory.max": f"{2 * GB}\n",
                "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{8 * GB}\n",
            },
        )
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

        assert m._read_cgroup_memory_limit() == 2 * GB

    def test_unlimited_everywhere_still_reports_no_limit(self, monkeypatch, tmp_path):
        root = fake_root(
            tmp_path,
            cgroup="0::/system.slice/hermes-agent.service\n",
            mountinfo=V2_MOUNTINFO,
            limits={
                "/sys/fs/cgroup/memory.max": "max\n",
                "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": "max\n",
            },
        )
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

        assert m._read_cgroup_memory_limit() is None
        assert m._resolve_tui_heap_mb() == 8192

    def test_absent_proc_still_falls_back_to_the_fixed_paths(self, monkeypatch, tmp_path):
        """macOS and Windows have no ``/proc``; the hierarchy roots are all there is."""
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", _no_proc_root(tmp_path))
        limit_file = tmp_path / "memory.max"
        limit_file.write_text(f"{6 * GB}\n", encoding="utf-8")
        monkeypatch.setattr(cgroup_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))

        assert m._read_cgroup_memory_limit() == 6 * GB


class TestReadCgroupMemoryLimit:
    def test_v2_max_is_unlimited(self, tmp_path):
        assert _read({V2: "max"}, tmp_path) is None


class TestResolveTuiHeapMb:
    def _resolve(self, limit_bytes):
        with mock.patch.object(m, "_read_cgroup_memory_limit", return_value=limit_bytes):
            return m._resolve_tui_heap_mb()

    def test_unconstrained_uses_default(self):
        assert self._resolve(None) == 8192


class TestNodeOptionsTokenMerge:
    """The _launch_tui token-merge block must add the sized cap unless the user
    already supplied one, and must preserve unrelated NODE_OPTIONS flags."""

    def _merge(self, node_options, limit_bytes):
        with mock.patch.object(m, "_read_cgroup_memory_limit", return_value=limit_bytes):
            tokens = node_options.split()
            if not any(t.startswith("--max-old-space-size=") for t in tokens):
                tokens.append(f"--max-old-space-size={m._resolve_tui_heap_mb()}")
            return " ".join(tokens)

    def test_unconstrained_empty(self):
        assert self._merge("", None) == "--max-old-space-size=8192"


    def test_preserves_other_flags(self):
        assert self._merge("--enable-source-maps", 4 * GB) == "--enable-source-maps --max-old-space-size=3072"
