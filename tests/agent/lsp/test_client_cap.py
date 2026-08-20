"""Tests for the concurrent language-server cap.

The idle reaper (``DEFAULT_IDLE_TIMEOUT``) bounds how *long* a client
survives; it does not bound how *many* exist at once.  Thirteen live
``typescript-language-server`` processes holding ~16 GiB is what put
pai-mac-mini into swap while every one of them was inside its idle
window, so the reaper could not have helped.  These cover the cap that
bounds the population itself.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

import pytest

from agent import cgroup_memory as cgroup_mod
from agent.lsp import manager as manager_mod
from agent.lsp.manager import (
    FALLBACK_CLIENT_CAP,
    LSP_CLIENT_FOOTPRINT_BYTES,
    MAX_CLIENT_CAP,
    MIN_CLIENT_CAP,
    LSPService,
    default_max_clients,
)

GIB = 1024 * 1024 * 1024


class FakeClient:
    """Stands in for ``LSPClient``: identity plus an awaitable shutdown."""

    def __init__(self, server_id: str, workspace_root: str) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def make_service(
    max_clients: Optional[int] = None,
    idle_timeout: float = 600.0,
    memory_budget: object = None,
) -> LSPService:
    """A service with no background loop and no real subprocesses.

    ``enabled=False`` keeps ``_BackgroundLoop`` unstarted and the reaper
    unscheduled, so the eviction coroutines can be driven directly.

    The byte budget is DISABLED by default so these exercise the count
    cap and nothing else.  Left deriving from host RAM, they were
    host-dependent: a 1 GiB worker yields a 256 MiB budget, every fake
    ``pyright`` client is charged 800 MiB, and the byte bound evicts to
    the floor before the count cap under test can bind — three tests
    here failed on such a host while passing on a 16 GiB one.  The byte
    half has its own coverage in ``test_client_footprint.py``.
    """
    return LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=idle_timeout,
        max_clients=max_clients,
        memory_budget=memory_budget,
    )


@pytest.fixture
def running_service():
    """Build services with a REAL background loop, and stop them after.

    ``make_service`` uses ``enabled=False``, which never starts the
    loop — fine for driving the eviction coroutines by hand, useless
    for the release path, whose whole point is that nothing hand-cranks
    it.  These tests must go through the production scheduling path.
    """
    services = []

    def _make(
        max_clients: Optional[int] = None,
        idle_timeout: float = 0.0,
        memory_budget: object = None,
    ) -> LSPService:
        svc = LSPService(
            enabled=True,
            wait_mode="document",
            wait_timeout=1.0,
            install_strategy="manual",
            idle_timeout=idle_timeout,
            max_clients=max_clients,
            memory_budget=memory_budget,
        )
        services.append(svc)
        return svc

    yield _make

    for svc in services:
        svc._loop.stop()


def drain_to(svc: LSPService, expected: int, timeout: float = 5.0) -> None:
    """Block until the fleet reaches *expected* clients, or give up.

    Polls rather than sleeping a fixed interval so a slow machine does
    not turn a correct fix into a flake, and a broken one still fails
    within the budget instead of hanging the suite.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(svc._clients) != expected:
        time.sleep(0.01)


def seed(svc: LSPService, *specs: Tuple[str, float]) -> dict:
    """Install fake clients keyed ``(pyright, <root>)`` with a last-used stamp."""
    clients = {}
    for root, last_used in specs:
        key = ("pyright", root)
        client = FakeClient("pyright", root)
        svc._clients[key] = client
        svc._last_used[key] = last_used
        clients[key] = client
    return clients


# ----------------------------------------------------------------------
# cap derivation
# ----------------------------------------------------------------------


def test_count_cap_does_not_bind_a_cheap_fleet():
    """The count ceiling is sized off the cheapest server, not typescript.

    Sizing it off typescript is what produced a cap of 3 on the 16 GiB
    host while the measured healthy working set was 7 (SCA-4688), even
    when every live server was a 41 MiB yaml process.  The memory budget
    is what actually holds the fleet down; see the byte-budget tests.
    """
    assert default_max_clients(16 * GIB) > 7


def test_cap_scales_with_a_larger_host():
    cheapest = min(
        [
            LSP_CLIENT_FOOTPRINT_BYTES,
            *manager_mod.LSP_SERVER_FOOTPRINT_BYTES.values(),
        ]
    )
    expected = min(MAX_CLIENT_CAP, (64 * GIB // 4) // cheapest)
    assert default_max_clients(64 * GIB) == expected


def test_cap_never_drops_below_one():
    """A cap of 0 would disable LSP outright rather than bound it."""
    assert default_max_clients(64 * 1024 * 1024) == MIN_CLIENT_CAP


def test_cap_is_ceilinged_on_a_very_large_host():
    """Headroom to hide unbounded growth is not a reason to allow it."""
    assert default_max_clients(4096 * GIB) == MAX_CLIENT_CAP


def test_unreadable_host_memory_uses_the_conservative_default(monkeypatch):
    """Under-caching costs a respawn; over-caching costs gigabytes."""
    monkeypatch.setattr(manager_mod, "host_memory_bytes", lambda: None)
    assert default_max_clients() == FALLBACK_CLIENT_CAP
    assert default_max_clients(0) == FALLBACK_CLIENT_CAP


def test_cgroup_limit_wins_over_node_memory_when_smaller(monkeypatch, tmp_path):
    """Inside a 4 GiB container, ``SC_PHYS_PAGES`` still reports the node's
    RAM; deriving the cap from that permits a population that OOMs."""
    limit_file = tmp_path / "memory.max"
    limit_file.write_text(str(4 * GIB), encoding="utf-8")
    no_proc(monkeypatch, tmp_path)
    monkeypatch.setattr(cgroup_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))
    monkeypatch.setattr(manager_mod.os, "sysconf", lambda name: (64 * GIB) // 4096 if "PHYS" in name else 4096)

    assert manager_mod.host_memory_bytes() == 4 * GIB


def test_cgroup_unlimited_sentinel_is_ignored(monkeypatch, tmp_path):
    """cgroup v1 writes a page-aligned LONG_MAX for "no limit"."""
    limit_file = tmp_path / "memory.limit_in_bytes"
    limit_file.write_text(str((1 << 63) - 4096), encoding="utf-8")
    no_proc(monkeypatch, tmp_path)
    monkeypatch.setattr(cgroup_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))

    assert cgroup_mod.cgroup_memory_limit_bytes() is None


# ----------------------------------------------------------------------
# cgroup path resolution
#
# The two tests above monkeypatch ``CGROUP_MEMORY_LIMIT_PATHS`` to a temp
# file, which proves the parsing and the min(), and can say nothing about
# *which file* a deployed process reads.  The fixed paths are hierarchy
# roots: under a systemd unit with ``MemoryMax=``, or in a container
# without a private cgroup namespace, the limit that binds the process
# lives at the path named in ``/proc/self/cgroup``, while the root reads
# ``max``.  These build a whole fake ``/proc`` + cgroupfs so the path
# resolution itself is the thing under test.
# ----------------------------------------------------------------------


V2_MOUNTINFO = (
    "31 24 0:27 {mount_root} {mount_point} rw,nosuid,nodev,noexec,relatime"
    " shared:9 - cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n"
)
V1_MOUNTINFO = (
    "24 23 0:21 / /sys/fs/cgroup ro,nosuid,nodev,noexec - tmpfs tmpfs ro,mode=755\n"
    "31 24 0:27 {mount_root} {mount_point} rw,nosuid,nodev,noexec,relatime"
    " - cgroup cgroup rw,memory\n"
)


def no_proc(monkeypatch, tmp_path) -> None:
    """Point the resolver at a root with no ``/proc``, as on macOS.

    Without this the tests that exercise the fixed-path fallback would
    read the *real* ``/proc`` on Linux CI and stop being deterministic.
    """
    empty = tmp_path / "no-proc-root"
    empty.mkdir(exist_ok=True)
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", str(empty))


def fake_root(
    tmp_path,
    *,
    cgroup: str,
    mountinfo: str,
    limits: dict,
) -> str:
    """Materialise a fake filesystem root and return its path.

    *cgroup* is the ``/proc/self/cgroup`` body, *mountinfo* the
    ``/proc/self/mountinfo`` body, and *limits* maps an absolute
    cgroupfs path to the text of the limit file at it.
    """
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


def test_v2_limit_read_from_this_process_cgroup_not_the_hierarchy_root(monkeypatch, tmp_path):
    """A systemd unit's ``MemoryMax=`` lives under its own cgroup path.

    The hierarchy root reads ``max``.  Reading only the root reports
    "unlimited", falls through to ``SC_PHYS_PAGES``, and sizes the fleet
    off the node's 64 GiB while the unit may use 4 GiB.
    """
    root = fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=V2_MOUNTINFO.format(mount_root="/", mount_point="/sys/fs/cgroup"),
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{4 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() == 4 * GIB


def test_v2_node_memory_does_not_size_the_cap_inside_a_constrained_unit(monkeypatch, tmp_path):
    """The consequence: a node-sized fleet against a unit-sized budget.

    Written when one flat footprint charged every server type, where the
    whole bound was a count and the unit afforded exactly 1.  Admission is
    now two bounds (SCA-4688), so the same defect has to be stated against
    both halves: the count ceiling must still come from the unit's 4 GiB
    rather than the node's 64 GiB, and the byte budget it is paired with
    must still be the unit's.
    """
    root = fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=V2_MOUNTINFO.format(mount_root="/", mount_point="/sys/fs/cgroup"),
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{4 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)
    monkeypatch.setattr(
        manager_mod.os, "sysconf", lambda name: (64 * GIB) // 4096 if "PHYS" in name else 4096
    )

    assert manager_mod.host_memory_bytes() == 4 * GIB

    # The count half.  Reading the node would clamp at MAX_CLIENT_CAP; the
    # unit's 4 GiB affords a quarter of that, so the cap is still derived
    # from the constrained unit and not from the node behind it.
    assert manager_mod.default_max_clients(64 * GIB) == MAX_CLIENT_CAP
    assert manager_mod.default_max_clients() == 6

    # The memory half, which is what actually holds the fleet inside the
    # unit now that the count ceiling divides by the cheapest server: the
    # unit affords 1 GiB, which one typescript server already exceeds.
    budget = manager_mod.memory_budget_bytes()
    assert budget == 1024 * 1024 * 1024
    assert budget < manager_mod.footprint_for("typescript")


def test_an_ancestor_limit_binds_a_descendant(monkeypatch, tmp_path):
    """A slice capped below its child's own cap is the one that binds.

    The leaf may declare 8 GiB; if the enclosing slice allows 2 GiB the
    process still gets 2 GiB, so the walk takes the minimum rather than
    the first limit it finds.
    """
    root = fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=V2_MOUNTINFO.format(mount_root="/", mount_point="/sys/fs/cgroup"),
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": f"{2 * GIB}\n",
            "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{8 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() == 2 * GIB


def test_v2_unlimited_everywhere_reports_no_limit(monkeypatch, tmp_path):
    """An unconstrained host must still fall through to node memory."""
    root = fake_root(
        tmp_path,
        cgroup="0::/user.slice/user-1000.slice\n",
        mountinfo=V2_MOUNTINFO.format(mount_root="/", mount_point="/sys/fs/cgroup"),
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/user-1000.slice/memory.max": "max\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() is None


def test_v1_memory_limit_read_from_the_controller_mount(monkeypatch, tmp_path):
    """v1 puts the memory controller on its own mount, path from column 3."""
    root = fake_root(
        tmp_path,
        cgroup="9:memory:/docker/deadbeef\n2:cpu,cpuacct:/docker/deadbeef\n",
        mountinfo=V1_MOUNTINFO.format(
            mount_root="/", mount_point="/sys/fs/cgroup/memory"
        ),
        limits={
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": f"{(1 << 63) - 4096}\n",
            "/sys/fs/cgroup/memory/docker/deadbeef/memory.limit_in_bytes": f"{3 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() == 3 * GIB


def test_mount_root_prefix_is_stripped_from_the_cgroup_path(monkeypatch, tmp_path):
    """A bind-mounted subtree makes the cgroup path outer, the mount inner.

    ``/proc/self/cgroup`` names ``/system.slice/hermes-agent.service``
    while the mount exposes ``/system.slice`` at ``/sys/fs/cgroup``, so
    the directory on disk is ``/sys/fs/cgroup/hermes-agent.service``.
    Joining the raw cgroup path onto the mount point would miss it.
    """
    root = fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=V2_MOUNTINFO.format(
            mount_root="/system.slice", mount_point="/sys/fs/cgroup"
        ),
        limits={
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/hermes-agent.service/memory.max": f"{5 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() == 5 * GIB


def test_a_bind_mount_listed_first_does_not_shadow_the_real_hierarchy(monkeypatch, tmp_path):
    """Several cgroup2 mounts can exist; the first is not always ours.

    A bind mount rooted outside this process's cgroup cannot map its
    path at all.  Accepting the first mountinfo entry reads that
    unrelated mount's root, finds it unlimited, and reports "no limit"
    without ever consulting the full hierarchy that holds the real one.
    """
    root = fake_root(
        tmp_path,
        cgroup="0::/system.slice/hermes-agent.service\n",
        mountinfo=(
            V2_MOUNTINFO.format(mount_root="/other.slice", mount_point="/sys/fs/cgroup/bind")
            + V2_MOUNTINFO.format(mount_root="/", mount_point="/sys/fs/cgroup")
        ),
        limits={
            "/sys/fs/cgroup/bind/memory.max": "max\n",
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/hermes-agent.service/memory.max": f"{4 * GIB}\n",
        },
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)

    assert cgroup_mod.cgroup_memory_limit_bytes() == 4 * GIB


def test_absent_proc_falls_back_to_the_fixed_paths(monkeypatch, tmp_path):
    """macOS and Windows have no ``/proc``; the old behaviour is the floor."""
    limit_file = tmp_path / "memory.max"
    limit_file.write_text(str(6 * GIB), encoding="utf-8")
    no_proc(monkeypatch, tmp_path)
    monkeypatch.setattr(cgroup_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))

    assert cgroup_mod.cgroup_memory_limit_bytes() == 6 * GIB


def test_unparseable_proc_cgroup_falls_back_to_the_fixed_paths(monkeypatch, tmp_path):
    """Garbage in ``/proc/self/cgroup`` must not lose the root reading."""
    root = fake_root(
        tmp_path,
        cgroup="not a cgroup line\n",
        mountinfo="junk\n",
        limits={"/sys/fs/cgroup/memory.max": f"{7 * GIB}\n"},
    )
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS_ROOT", root)
    monkeypatch.setattr(
        cgroup_mod,
        "CGROUP_MEMORY_LIMIT_PATHS",
        (str(tmp_path / "root" / "sys" / "fs" / "cgroup" / "memory.max"),),
    )

    assert cgroup_mod.cgroup_memory_limit_bytes() == 7 * GIB


# ----------------------------------------------------------------------
# cap enforcement
# ----------------------------------------------------------------------


def test_cap_evicts_least_recently_used_first():
    svc = make_service(max_clients=2)
    clients = seed(svc, ("/oldest", 100.0), ("/middle", 200.0), ("/newest", 300.0))

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == [("pyright", "/oldest")]
    assert set(svc._clients) == {("pyright", "/middle"), ("pyright", "/newest")}
    assert clients[("pyright", "/oldest")].shutdown_calls == 1


def test_cap_drains_until_satisfied_not_just_one():
    svc = make_service(max_clients=1)
    seed(svc, ("/a", 100.0), ("/b", 200.0), ("/c", 300.0))

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == [("pyright", "/a"), ("pyright", "/b")]
    assert set(svc._clients) == {("pyright", "/c")}


def test_cap_never_evicts_the_client_that_just_spawned():
    """Evicting the caller's own client turns a cap into an infinite
    spawn/evict loop: it respawns, re-triggers the cap, and repeats."""
    svc = make_service(max_clients=1)
    seed(svc, ("/old", 500.0), ("/fresh", 100.0))
    protect = ("pyright", "/fresh")

    evicted = asyncio.run(svc._enforce_cap_async(protect=protect))

    assert evicted == [("pyright", "/old")]
    assert protect in svc._clients


def test_cap_never_evicts_a_client_that_has_not_reached_its_caller_yet():
    """A concurrent spawn must not evict another spawn's fresh client.

    ``_get_or_spawn`` inserts into ``_clients`` and only then returns;
    the caller runs ``_acquire`` *after* that return.  In the window
    between, the client is live but its in-flight count is still zero,
    so ``_evictable`` offers it up.  ``protect`` cannot cover it — that
    is per-call, and the sweep doing the evicting belongs to a
    *different* root's spawn, which passes no protect at all.

    The victim's caller then receives a client that has already been
    shut down and silently loses its diagnostics.  A key still
    registered in ``_spawning`` has not been handed over yet and must
    be off the table.
    """
    svc = make_service(max_clients=1)
    seed(svc, ("/fresh", 100.0))
    fresh = ("pyright", "/fresh")
    # ``/fresh`` finished starting but its caller has not acquired it.
    # Only the *keys* of ``_spawning`` matter to the cap; the futures
    # are awaited solely by callers that join an in-progress spawn.
    svc._spawning[fresh] = None
    # A second root begins spawning and sweeps with no ``protect``.
    svc._spawning[("pyright", "/other")] = None

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == []
    assert fresh in svc._clients
    assert svc._clients[fresh].shutdown_calls == 0


def test_cap_does_not_evict_a_client_with_a_request_in_flight():
    """Reaping mid-flight makes the outer wait time out, and the handler
    marks that (server, workspace) pair broken for the whole process
    lifetime — the cap must not be able to cause that."""
    svc = make_service(max_clients=1)
    seed(svc, ("/busy", 100.0), ("/idle", 200.0))
    svc._acquire(("pyright", "/busy"))

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == [("pyright", "/idle")]
    assert ("pyright", "/busy") in svc._clients


def test_cap_tolerates_every_client_being_busy():
    """Going over the cap briefly beats killing a live request."""
    svc = make_service(max_clients=1)
    seed(svc, ("/a", 100.0), ("/b", 200.0))
    svc._acquire(("pyright", "/a"))
    svc._acquire(("pyright", "/b"))

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == []
    assert len(svc._clients) == 2


def test_release_makes_a_client_evictable_again():
    svc = make_service(max_clients=1)
    seed(svc, ("/a", 100.0), ("/b", 200.0))
    key = ("pyright", "/a")
    svc._acquire(key)
    svc._release(key)

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == [key]


# ----------------------------------------------------------------------
# the escape hatch closes itself: release re-enforces
# ----------------------------------------------------------------------


def test_release_drains_the_overage_with_nothing_hand_cranking_it(running_service):
    """The all-busy break leaves the fleet over cap.  Nothing else closes it.

    ``idle_timeout=0`` is a documented setting for keeping indexes
    warm, and it disables the reaper outright — so the reaper cannot be
    the backstop the break comment assumes.  The spawn path only runs
    on the next spawn, which a saturated burst may never issue.  This
    asserts the fleet returns to the cap on releases alone, with no
    ``enforce_cap_now()`` anywhere: that hand-crank is what let
    ``test_release_makes_a_client_evictable_again`` pass while the
    production lifecycle stayed broken.
    """
    svc = running_service(max_clients=2, idle_timeout=0.0)
    seed(svc, ("/a", 100.0), ("/b", 200.0), ("/c", 300.0), ("/d", 400.0))
    keys = [("pyright", root) for root in ("/a", "/b", "/c", "/d")]
    for key in keys:
        svc._acquire(key)

    # Precondition: this is the escape hatch, not a quiet no-op.
    assert svc._loop.run(svc._enforce_cap_async(), timeout=5.0) == []
    assert len(svc._clients) == 4

    for key in keys:
        svc._release(key)

    drain_to(svc, 2)

    assert set(svc._clients) == {("pyright", "/c"), ("pyright", "/d")}


def test_release_triggered_sweep_still_refuses_to_evict_a_busy_client(running_service):
    """The anti-criterion, re-asserted on the new path.

    Closing the hatch must not be done by reaping mid-request: that
    times out the outer diagnostics wait and marks the workspace broken
    for the process lifetime.
    """
    svc = running_service(max_clients=1, idle_timeout=0.0)
    seed(svc, ("/busy", 100.0), ("/done", 200.0))
    busy = ("pyright", "/busy")
    done = ("pyright", "/done")
    svc._acquire(busy)
    svc._acquire(done)

    svc._release(done)
    drain_to(svc, 1)

    assert set(svc._clients) == {busy}


def test_release_under_the_cap_schedules_no_sweep(monkeypatch):
    """Every request ends in a release; only the over-cap ones may pay."""
    svc = make_service(max_clients=4)
    seed(svc, ("/a", 100.0), ("/b", 200.0))
    calls = []
    monkeypatch.setattr(svc, "_schedule_cap_enforcement", lambda: calls.append(1))
    key = ("pyright", "/a")

    svc._acquire(key)
    svc._release(key)

    assert calls == []


def test_partial_release_of_a_shared_client_schedules_no_sweep(monkeypatch):
    """A client with a second request still in flight is not evictable,
    so sweeping on the first release is pure churn."""
    svc = make_service(max_clients=1)
    seed(svc, ("/shared", 100.0), ("/other", 200.0))
    calls = []
    monkeypatch.setattr(svc, "_schedule_cap_enforcement", lambda: calls.append(1))
    key = ("pyright", "/shared")

    svc._acquire(key)
    svc._acquire(key)
    svc._release(key)
    assert calls == []

    svc._release(key)
    assert calls == [1]


def test_inflight_is_refcounted_for_concurrent_requests():
    """Two edits in one project share a client; the first to finish must
    not expose it while the second is still waiting."""
    svc = make_service(max_clients=1)
    seed(svc, ("/shared", 100.0), ("/other", 200.0))
    key = ("pyright", "/shared")
    svc._acquire(key)
    svc._acquire(key)
    svc._release(key)

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == [("pyright", "/other")]
    assert key in svc._clients


# ----------------------------------------------------------------------
# the reaper must respect the same in-flight guard
# ----------------------------------------------------------------------


def test_idle_reaper_does_not_reap_a_busy_client(monkeypatch):
    """``idle_timeout`` is clamped to 30s precisely because reaping
    mid-flight is unrecoverable; the in-flight guard closes the residual
    race that the clamp only makes unlikely."""
    svc = make_service(max_clients=8, idle_timeout=30.0)
    seed(svc, ("/busy", 0.0), ("/stale", 0.0))
    svc._acquire(("pyright", "/busy"))
    monkeypatch.setattr(manager_mod.time, "time", lambda: 10_000.0)

    asyncio.run(svc._reap_idle_once())

    assert ("pyright", "/busy") in svc._clients
    assert ("pyright", "/stale") not in svc._clients


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (4, 4),
        ("4", 4),
        (0, "derived"),          # 0 must not disable the bound
        (-1, "derived"),
        ("nonsense", "derived"),
        (float("inf"), "derived"),
        (float("nan"), "derived"),
    ],
)
def test_config_max_clients_falls_back_to_the_derived_cap_on_garbage(raw, expected):
    """Garbage must not silently restore unbounded accumulation."""
    resolved = manager_mod.resolve_max_clients(raw)
    if expected == "derived":
        assert resolved == default_max_clients()
    else:
        assert resolved == expected


def test_config_max_clients_none_means_derive_from_the_host():
    assert manager_mod.resolve_max_clients(None) == default_max_clients()


def test_config_max_clients_is_clamped_to_the_ceiling():
    assert manager_mod.resolve_max_clients(10_000) == MAX_CLIENT_CAP


# ----------------------------------------------------------------------
# eviction handoff: keeping a wedged shutdown off the caller's budget
# ----------------------------------------------------------------------


class WedgedClient(FakeClient):
    """A client whose shutdown never finishes until released."""

    def __init__(self, server_id: str, workspace_root: str) -> None:
        super().__init__(server_id, workspace_root)
        self.released = asyncio.Event()
        self.finished = False

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        await self.released.wait()
        self.finished = True


def _wedge(svc: LSPService, root: str, last_used: float) -> WedgedClient:
    key = ("pyright", root)
    client = WedgedClient("pyright", root)
    svc._clients[key] = client
    svc._last_used[key] = last_used
    return client


def test_a_wedged_shutdown_does_not_hold_the_handoff_caller():
    """With a handoff bound, the caller leaves; the shutdown keeps going.

    This is the spawn path's contract: cap enforcement runs inside the
    request's diagnostics budget, so it must be able to give up waiting
    on a victim without giving up on killing it.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        victim = _wedge(svc, "/wedged", 100.0)
        seed(svc, ("/fresh", 300.0))

        evicted = await svc._enforce_cap_async(handoff=0.05)

        assert evicted == [("pyright", "/wedged")]
        assert victim.shutdown_calls == 1, "the shutdown must still have been started"
        assert not victim.finished, "the caller waited out the wedged shutdown"

        # Still on the hook: releasing it lets the drain complete.
        victim.released.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert victim.finished, "the detached shutdown was abandoned, not drained"

    asyncio.run(scenario())


def test_an_in_flight_shutdown_keeps_occupying_its_slot():
    """A detached shutdown must not read as a free slot.

    The process is still resident until it exits, and the cap bounds
    resident processes.  ``_spawning`` reserves the slot of a server
    that is about to exist; this is the same reservation for one that
    is about to stop.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        victim = _wedge(svc, "/wedged", 100.0)
        seed(svc, ("/fresh", 300.0))

        await svc._enforce_cap_async(handoff=0.05)

        # Keyed by task; the workspace is the value (see the map's note).
        assert ("pyright", "/wedged") in svc._shutting_down.values()
        with svc._state_lock:
            # One live client plus one process still going down, against
            # a cap of one: the fleet is not back under the cap yet and
            # the accounting must say so.
            assert svc._overage_locked() == 1

        victim.released.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with svc._state_lock:
            assert svc._overage_locked() == 0, (
                "the reservation was never released once the process exited"
            )

    asyncio.run(scenario())


def test_the_sweep_stops_once_enough_shutdowns_are_in_flight():
    """Reserved slots must not drive the sweep into over-draining.

    With in-flight shutdowns counted against the cap, the overage does
    not fall when a victim is detached.  A sweep that only watched the
    overage would keep picking victims and take the whole fleet down.
    """
    async def scenario():
        svc = make_service(max_clients=2)
        _wedge(svc, "/oldest", 100.0)
        seed(svc, ("/middle", 200.0), ("/newest", 300.0))

        evicted = await svc._enforce_cap_async(handoff=0.05)

        assert evicted == [("pyright", "/oldest")], (
            "exactly one eviction covers an overage of one — the sweep "
            "kept going against reserved slots"
        )
        assert set(svc._clients) == {("pyright", "/middle"), ("pyright", "/newest")}

    asyncio.run(scenario())


def test_a_failing_shutdown_releases_its_slot_and_is_logged(caplog):
    """A shutdown that raises must not be silent, and must not leak a slot.

    Nobody calls ``.result()`` on the drain task.  If the exception
    vanished, a raising shutdown would hold its reservation forever and
    permanently shrink the usable fleet.
    """
    class ExplodingClient(FakeClient):
        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            raise RuntimeError("shutdown blew up")

    async def scenario():
        svc = make_service(max_clients=1)
        key = ("pyright", "/boom")
        svc._clients[key] = ExplodingClient("pyright", "/boom")
        svc._last_used[key] = 100.0
        seed(svc, ("/fresh", 300.0))

        with caplog.at_level("DEBUG", logger="agent.lsp.manager"):
            await svc._enforce_cap_async(handoff=0.05)

        assert key not in svc._shutting_down.values(), (
            "a failed shutdown leaked its slot"
        )
        assert any("shutdown blew up" in r.getMessage() for r in caplog.records), (
            "the shutdown failure was swallowed"
        )

    asyncio.run(scenario())


def test_the_hand_crank_still_waits_for_the_shutdown():
    """``handoff=None`` keeps the original await-to-completion contract.

    ``enforce_cap_now`` and the detached release sweep are not on any
    request's critical path, so they have no reason to hand off — and
    the CLI reporting an eviction that has not happened yet would be a
    lie.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        clients = seed(svc, ("/old", 100.0), ("/new", 300.0))

        await svc._enforce_cap_async()

        assert clients[("pyright", "/old")].shutdown_calls == 1
        assert not svc._shutting_down, (
            "the reservation must be gone once the shutdown completed"
        )

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# Codex review on PR #66: the handoff bound and the reservation map
# ----------------------------------------------------------------------


class HangingClient(FakeClient):
    """A victim that never finishes shutting down.

    The wedged case is the only one where the handoff bound is load
    bearing — a client that exits promptly never reaches it.
    """

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        await asyncio.Event().wait()


def _seed_hanging(svc: LSPService, *roots: str) -> dict:
    clients = {}
    for index, root in enumerate(roots):
        key = ("pyright", root)
        client = HangingClient("pyright", root)
        svc._clients[key] = client
        svc._last_used[key] = float(index)
        clients[key] = client
    return clients


async def _cancel_detached(svc: LSPService) -> None:
    """Drop the detached drains so the loop closes without warnings."""
    tasks = list(svc._shutting_down)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_handoff_bounds_the_whole_sweep_not_each_victim():
    """One deadline for the sweep, not ``handoff`` per wedged victim.

    *handoff* is the slice of its request budget the caller is willing to
    spend making room.  Charged per victim it scaled with fleet overage:
    four wedged victims spent 4 x 0.25s here, so ``get_diagnostics_sync``
    could still time out before ``wait_for_diagnostics`` ever got its
    configured budget — the silent no-diagnostics result the bound exists
    to prevent.

    Asserts wall clock rather than call counts because the defect IS
    elapsed time: a per-victim implementation passes any assertion about
    which clients were evicted.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        _seed_hanging(svc, "/a", "/b", "/c", "/d", "/e")

        start = time.monotonic()
        evicted = await svc._enforce_cap_async(handoff=0.25)
        elapsed = time.monotonic() - start

        # Four victims, every one of them wedged.
        assert len(evicted) == 4
        # Per-victim would be 4 x 0.25 = 1.0s.  The midpoint discriminates
        # without being tight enough to flake on a loaded CI box.
        assert elapsed < 0.6, f"sweep spent {elapsed:.2f}s; budget was 0.25s"

        await _cancel_detached(svc)

    asyncio.run(scenario())


def test_sweep_past_its_deadline_still_evicts_at_zero_wait():
    """Budget exhaustion must not stop the sweep leaving the fleet over cap.

    Past the deadline the remaining victims detach immediately with their
    slots still reserved — the same trade ``_evict`` already makes for a
    single wedged victim, applied to the tail of the sweep.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        _seed_hanging(svc, "/a", "/b", "/c")

        evicted = await svc._enforce_cap_async(handoff=0.05)

        assert len(evicted) == 2
        assert set(svc._clients) == {("pyright", "/c")}
        # Every detached drain still holds its slot, so the cap sees the
        # processes that are genuinely still resident.
        assert len(svc._shutting_down) == 2

        await _cancel_detached(svc)

    asyncio.run(scenario())


def test_concurrent_shutdowns_of_one_workspace_keep_separate_reservations():
    """Evict, respawn the same root, evict again before the first drains.

    ``_shutting_down`` is keyed by task rather than by
    ``(server_id, workspace_root)`` precisely for this: every reader of
    that map counts *processes*, and one workspace can have two resident
    at once.  Under a per-key dict the second registration overwrote the
    first, and the first drain's ``finally`` then removed the second's
    entry — so the replacement process was neither counted against the
    cap nor awaited at service stop.
    """
    async def scenario():
        svc = make_service(max_clients=1)
        key = ("pyright", "/same")

        # First victim exits promptly; its drain is what used to evict the
        # replacement's reservation on the way out.
        first = FakeClient(*key)
        svc._clients[key] = first
        svc._last_used[key] = 100.0
        assert await svc._evict(key, "first", handoff=0.0)

        # The replacement is spawned and evicted before that drain lands.
        second = HangingClient(*key)
        svc._clients[key] = second
        svc._last_used[key] = 200.0
        assert await svc._evict(key, "second", handoff=0.0)

        assert len(svc._shutting_down) == 2, (
            "two resident processes for one workspace must hold two slots"
        )

        # Let the first drain finish and release ONLY its own reservation.
        for _ in range(100):
            if first.shutdown_calls and len(svc._shutting_down) == 1:
                break
            await asyncio.sleep(0.01)

        assert len(svc._shutting_down) == 1, (
            "the completing drain must not release the replacement's slot"
        )
        # The survivor is the still-wedged replacement, not the finished one.
        assert list(svc._shutting_down.values()) == [key]
        assert svc._overage_locked() >= 0

        await _cancel_detached(svc)

    asyncio.run(scenario())


def test_count_cap_behaviour_is_independent_of_host_memory(monkeypatch):
    """These tests must not change verdict with the worker's RAM.

    The byte budget derived from host memory unconditionally, so this
    suite silently became host-dependent: on a 1 GiB worker the budget is
    256 MiB, every fake ``pyright`` client is charged 800 MiB, and the
    byte bound evicts down to the floor before the count cap under test
    can bind — ``test_cap_evicts_least_recently_used_first`` and two
    others failed there while passing on a 16 GiB host.  The fixtures now
    pin the byte half off, so a count-cap test measures the count cap.
    """
    monkeypatch.setattr(manager_mod, "host_memory_bytes", lambda: 1 * GIB)
    svc = make_service(max_clients=4)

    assert svc._memory_budget is None
    seed(svc, ("/a", 100.0), ("/b", 200.0), ("/c", 300.0), ("/d", 400.0))

    evicted = asyncio.run(svc._enforce_cap_async())

    assert evicted == []
    assert len(svc._clients) == 4


def test_the_byte_budget_can_still_be_pinned_explicitly():
    """Disabling by default must not make the byte half untestable."""
    svc = make_service(max_clients=24, memory_budget=1024 * 1024 * 1024)
    assert svc._memory_budget == 1024 * 1024 * 1024
