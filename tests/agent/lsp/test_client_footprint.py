"""Tests for per-type footprint accounting in the language-server cap.

Charging every server type the typescript figure derived a cap of 3 on the
16 GiB host this fleet runs on, against a measured healthy peak concurrency
of 7 — so ordinary multi-worktree work evicted and respawned continuously
(SCA-4688).  The fix charges each ``server_id`` its own measured footprint
against the same memory budget.

The footprint values under test come from ``scripts/measure_lsp_footprint.py``;
the raw readings are in ``docs/lsp-footprint-measurements.json``.
"""
from __future__ import annotations

from typing import Optional

import pytest

from agent.lsp import manager as manager_mod
from agent.lsp.manager import (
    LSP_CLIENT_FOOTPRINT_BYTES,
    LSP_MEMORY_BUDGET_FRACTION,
    LSP_SERVER_FOOTPRINT_BYTES,
    LSPService,
    footprint_for,
    memory_budget_bytes,
)

GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024

# The host in the defect report, and the host the fleet actually runs on.
INCIDENT_HOST_BYTES = 16 * GIB


class FakeClient:
    def __init__(self, server_id: str, workspace_root: str) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root

    async def shutdown(self) -> None:  # pragma: no cover - not exercised here
        return None


def make_service(total_bytes: int = INCIDENT_HOST_BYTES) -> LSPService:
    """A service whose memory budget is pinned to a known host size."""
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=600.0,
        max_clients=manager_mod.MAX_CLIENT_CAP,
    )
    service._memory_budget = memory_budget_bytes(total_bytes)
    return service


def admit(service: LSPService, fleet) -> int:
    """Install *fleet* as live clients; return the resulting memory overage.

    *fleet* is a list of ``server_id`` strings — each gets its own workspace
    root so every entry is a distinct client key, which is what a
    multi-worktree session actually produces.
    """
    for index, server_id in enumerate(fleet):
        key = (server_id, f"/w/{index}")
        service._clients[key] = FakeClient(*key)
        service._last_used[key] = float(index)
    with service._state_lock:
        return service._memory_overage_locked()


# ----------------------------------------------------------------------
# criterion 1 & 4: the table itself
# ----------------------------------------------------------------------


def test_measured_types_are_cheaper_than_the_flat_constant():
    """Every measured type costs less than the typescript-derived constant.

    If one did not, per-type accounting would be admitting *more* memory
    than the flat charge it replaced.
    """
    for server_id, cost in LSP_SERVER_FOOTPRINT_BYTES.items():
        if server_id == "typescript":
            continue
        assert cost < LSP_CLIENT_FOOTPRINT_BYTES, server_id


def test_typescript_is_charged_at_least_the_old_constant():
    """The measurement came in *above* 1.3 GiB, so it must not drop.

    1688 MiB measured against a large TypeScript project.  Charging the old
    flat 1300 would under-count the one type the original incident was
    actually about.
    """
    assert footprint_for("typescript") >= LSP_CLIENT_FOOTPRINT_BYTES


def test_unknown_server_charges_the_conservative_default():
    """Criterion 4: an unmeasured type is charged exactly as it is today.

    Absence from the table must never be a discount — rust-analyzer, gopls
    and jdtls are unmeasured *and* heavy.
    """
    assert footprint_for("rust-analyzer") == LSP_CLIENT_FOOTPRINT_BYTES
    assert footprint_for("not-a-real-server") == LSP_CLIENT_FOOTPRINT_BYTES


def test_budget_is_unchanged_by_per_type_accounting():
    """The ceiling did not move; only how servers are charged against it."""
    assert memory_budget_bytes(INCIDENT_HOST_BYTES) == int(
        INCIDENT_HOST_BYTES * LSP_MEMORY_BUDGET_FRACTION
    )


# ----------------------------------------------------------------------
# criterion 2: the measured working set fits
# ----------------------------------------------------------------------


def test_mixed_fleet_of_seven_fits_the_incident_host():
    """Criterion 2: the measured healthy peak of 7 is admitted.

    The real mix on this host is mostly pyright / yaml / bash with a
    typescript server or two.  Under the flat charge this fleet reported an
    overage of 4 and evicted continuously.
    """
    service = make_service()
    fleet = [
        "typescript",
        "pyright",
        "pyright",
        "yaml-language-server",
        "yaml-language-server",
        "bash-language-server",
        "bash-language-server",
    ]
    assert admit(service, fleet) == 0


def test_the_same_fleet_would_have_been_over_the_old_flat_charge():
    """Guards the premise: 7 servers x 1300 MiB really does exceed 4 GiB.

    Without this, the test above could pass because the budget is generous
    rather than because per-type accounting changed anything.
    """
    budget = memory_budget_bytes(INCIDENT_HOST_BYTES)
    assert 7 * LSP_CLIENT_FOOTPRINT_BYTES > budget


# ----------------------------------------------------------------------
# criterion 3: an expensive fleet still binds
# ----------------------------------------------------------------------


def test_typescript_heavy_fleet_still_binds():
    """Criterion 3: cheap servers fitting must not let expensive ones in.

    Four typescript servers is 6.6 GiB on a host whose editor-tooling
    budget is 4 GiB — the shape of the SCA-4389 outage.  It must report an
    overage rather than being admitted.
    """
    service = make_service()
    overage = admit(service, ["typescript"] * 4)
    assert overage > 0


def test_typescript_fleet_is_bounded_below_the_incident_footprint():
    """Whatever survives must cost less than the 25.81 GiB that broke the host."""
    service = make_service()
    fleet = ["typescript"] * 8
    overage = admit(service, fleet)
    survivors = len(fleet) - overage
    assert survivors * footprint_for("typescript") <= memory_budget_bytes(
        INCIDENT_HOST_BYTES
    )


def test_expensive_fleet_evicts_least_recently_used_first():
    """The overage count must match what the sweep can actually remove.

    ``_enforce_cap_async`` takes victims LRU-first, so an overage computed
    against any other order would either under- or over-evict.
    """
    service = make_service()
    # Oldest entries are the cheap ones; removing them frees little, so the
    # count has to reflect that rather than assuming it can drop a
    # typescript server first.
    fleet = ["yaml-language-server", "yaml-language-server", "typescript", "typescript", "typescript"]
    overage = admit(service, fleet)
    assert overage >= 3


# ----------------------------------------------------------------------
# floors and degraded hosts
# ----------------------------------------------------------------------


def test_a_single_oversized_server_is_never_evicted():
    """One typescript server on a 4 GiB host exceeds the budget outright.

    Evicting it would make every spawn immediately undo itself and the
    feature would never serve a single request.
    """
    service = make_service(4 * GIB)
    assert admit(service, ["typescript"]) == 0


def test_unreadable_host_memory_disables_the_byte_bound():
    """With no budget the count cap is the only bound, not a zero budget.

    A ``None`` budget treated as 0 would evict the entire fleet on a host
    whose memory simply could not be read.
    """
    service = make_service()
    service._memory_budget = None
    assert admit(service, ["typescript"] * 6) == 0


def test_shutting_down_servers_still_hold_their_memory():
    """An evicted server's pages are not back until its process is gone.

    ``_shutting_down`` is keyed by task with the client key as the value;
    charging the task would raise ``TypeError`` or silently charge nothing.
    """
    service = make_service()
    # Two cheap clients are far inside the budget on their own...
    assert admit(service, ["yaml-language-server", "yaml-language-server"]) == 0
    # ...until three draining typescript servers are charged alongside them.
    for index in range(3):
        service._shutting_down[object()] = ("typescript", f"/w/draining{index}")
    with service._state_lock:
        assert service._memory_overage_locked() > 0


# ----------------------------------------------------------------------
# review findings on PR #67: the two-part bound's enforcement path
# ----------------------------------------------------------------------


def test_pending_spawn_counts_against_the_one_client_floor():
    """One live server plus one pending one must report its overage.

    The floor exists so a single server larger than the whole budget is
    not evicted-and-respawned forever.  Applied to ``_clients`` alone it
    also silenced the *two*-server case: one live typescript plus one
    pending typescript is 3.4 GiB against a 1 GiB budget, but
    ``len(_clients) - MIN_CLIENT_CAP`` is zero, so nothing was removable
    and the overage read as zero.  ``_get_or_spawn`` sweeps BEFORE
    ``client.start()`` precisely to keep the fleet off that peak, and a
    zero overage let both processes index concurrently.

    Evicting the live one still leaves the pending one, so honouring the
    floor never required ignoring this.
    """
    service = make_service()
    service._memory_budget = 1024 * MIB
    live = ("typescript", "/w/live")
    service._clients[live] = FakeClient(*live)
    service._last_used[live] = 1.0
    # Reserved by a concurrent spawn; only the KEY matters to the cap.
    service._spawning[("typescript", "/w/pending")] = None

    with service._state_lock:
        assert service._memory_overage_locked() == 1


def test_a_lone_client_over_budget_is_still_never_evicted():
    """The floor itself must survive the fix above.

    One typescript server on a 1 GiB budget is over on its own.  Evicting
    it would respawn on the next request, forever, costing more than the
    memory it saves — with nothing pending, the answer is still zero.
    """
    service = make_service()
    service._memory_budget = 1024 * MIB
    lone = ("typescript", "/w/lone")
    service._clients[lone] = FakeClient(*lone)
    service._last_used[lone] = 1.0

    with service._state_lock:
        assert service._memory_overage_locked() == 0


def test_a_cheap_drain_does_not_satisfy_an_expensive_overage():
    """A draining 150 MiB server must not stop the sweep for 1.7 GiB ones.

    The old stop condition compared the NUMBER of drains in flight to the
    NUMBER of victims wanted.  Those units are interchangeable only while
    every server costs the same.  With one yaml server draining and three
    typescript servers live against a 4 GiB budget, one drain met a
    one-victim overage and the sweep stopped — then the cheap process
    exited and the three typescript servers sat 1 GiB over budget with no
    sweep scheduled to notice.

    Asked as "would we still be over once the drains land?", the answer is
    yes, so eviction must continue.
    """
    service = make_service()
    service._memory_budget = 4096 * MIB
    for index in range(3):
        key = ("typescript", f"/w/ts{index}")
        service._clients[key] = FakeClient(*key)
        service._last_used[key] = float(index)
    # The cheap victim is already detached and draining.
    service._shutting_down[object()] = ("yaml-language-server", "/w/yaml")

    with service._state_lock:
        residual = service._overage_locked(exclude_draining=True)
    # 3 x 1700 = 5100 MiB against 4096: the drain releases 150 MiB and
    # cannot close that, so a victim is still required.
    assert residual == 1


def test_byte_triggered_eviction_does_not_report_the_count_cap():
    """A byte-budget eviction must not log ``cap 24 exceeded``.

    On a 16 GiB host three typescript servers blow the 4 GiB budget while
    sitting far below the derived count cap of 24.  Reporting the count
    cap there tells an operator to look at a bound that is not binding.
    """
    service = make_service()
    for index in range(3):
        key = ("typescript", f"/w/ts{index}")
        service._clients[key] = FakeClient(*key)
        service._last_used[key] = float(index)

    with service._state_lock:
        overage, reason = service._binding_overage_locked()

    assert overage > 0
    assert "memory budget" in reason
    assert str(service._max_clients) not in reason


def test_count_triggered_eviction_still_reports_the_count_cap():
    """The count half must keep its own reason when it is the binding one."""
    service = make_service()
    service._memory_budget = None
    service._max_clients = 2
    for index in range(4):
        key = ("yaml-language-server", f"/w/y{index}")
        service._clients[key] = FakeClient(*key)
        service._last_used[key] = float(index)

    with service._state_lock:
        overage, reason = service._binding_overage_locked()

    assert overage == 2
    assert reason == "cap 2 exceeded"


def test_production_still_derives_the_byte_budget_from_the_host(monkeypatch):
    """Omitting ``memory_budget`` must read host memory, not disable it.

    The test fixtures pin the byte half OFF so a count-cap test measures
    the count cap.  That convenience must not be able to leak into
    production: flipping the constructor default from "derive" to ``None``
    would silently remove the bound that holds the fleet inside RAM, and
    every test would still pass.  The sentinel is what keeps "read the
    host" distinguishable from "this fleet has no byte budget".
    """
    monkeypatch.setattr(manager_mod, "host_memory_bytes", lambda: 16 * GIB)
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=600.0,
        max_clients=None,
    )
    assert service._memory_budget == memory_budget_bytes(16 * GIB)


def test_an_explicit_none_budget_disables_the_byte_bound(monkeypatch):
    """``None`` must mean disabled even on a host with readable memory."""
    monkeypatch.setattr(manager_mod, "host_memory_bytes", lambda: 16 * GIB)
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=600.0,
        max_clients=None,
        memory_budget=None,
    )
    assert service._memory_budget is None
    # ...and with no byte bound, a fleet that would blow the budget is
    # admitted, so the count cap is the only thing left holding it.
    assert admit(service, ["typescript"] * 6) == 0
