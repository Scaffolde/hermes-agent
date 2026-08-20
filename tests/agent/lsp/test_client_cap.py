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


def make_service(max_clients: Optional[int] = None, idle_timeout: float = 600.0) -> LSPService:
    """A service with no background loop and no real subprocesses.

    ``enabled=False`` keeps ``_BackgroundLoop`` unstarted and the reaper
    unscheduled, so the eviction coroutines can be driven directly.
    """
    return LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=idle_timeout,
        max_clients=max_clients,
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

    def _make(max_clients: Optional[int] = None, idle_timeout: float = 0.0) -> LSPService:
        svc = LSPService(
            enabled=True,
            wait_mode="document",
            wait_timeout=1.0,
            install_strategy="manual",
            idle_timeout=idle_timeout,
            max_clients=max_clients,
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


def test_cap_derived_from_host_memory_matches_the_incident_host():
    """16 GiB / 4 = one quarter budget; at ~1.3 GiB per server that is 3.

    This is the number that would have held pai-mac-mini: 3 servers
    rather than the 13 that were live when it swapped.
    """
    assert default_max_clients(16 * GIB) == 3


def test_cap_scales_with_a_larger_host():
    assert default_max_clients(64 * GIB) == (64 * GIB // 4) // LSP_CLIENT_FOOTPRINT_BYTES


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
    monkeypatch.setattr(manager_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))
    monkeypatch.setattr(manager_mod.os, "sysconf", lambda name: (64 * GIB) // 4096 if "PHYS" in name else 4096)

    assert manager_mod.host_memory_bytes() == 4 * GIB


def test_cgroup_unlimited_sentinel_is_ignored(monkeypatch, tmp_path):
    """cgroup v1 writes a page-aligned LONG_MAX for "no limit"."""
    limit_file = tmp_path / "memory.limit_in_bytes"
    limit_file.write_text(str((1 << 63) - 4096), encoding="utf-8")
    monkeypatch.setattr(manager_mod, "CGROUP_MEMORY_LIMIT_PATHS", (str(limit_file),))

    assert manager_mod._cgroup_memory_limit_bytes() is None


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
