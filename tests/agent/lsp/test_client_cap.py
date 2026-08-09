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
