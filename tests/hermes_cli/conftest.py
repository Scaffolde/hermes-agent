"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )


@pytest.fixture(autouse=True)
def _restore_dashboard_auth_required():
    """Snapshot and restore ``web_server.app.state.auth_required`` per test.

    ``app`` is a module-level FastAPI singleton, so ``app.state`` is
    process-global. The dashboard-auth gate tests drive ``start_server`` into
    its public-bind path and leave ``auth_required`` True (they assert on it and
    never put it back). Every later test that reuses ``app`` then hits the gated
    branch and gets 401 — ``test_dashboard_param_clamps.py`` alone loses 9 tests
    that way, and the whole-tree run loses 138+ across 57 files (SCA-4692).

    CI never saw it because those files are pinned to one xdist worker, so the
    leak stayed inside that worker: the green was a property of the sharding,
    not of the code. Restoring here fixes the leak at its source rather than
    codifying the partition.

    Deliberately does NOT import web_server — only restores it when a test (or
    its module) has already imported it, so this stays free for the ~490 files
    that never touch the dashboard.
    """
    module = sys.modules.get("hermes_cli.web_server")
    if module is None:
        yield
        return

    missing = object()
    previous = getattr(module.app.state, "auth_required", missing)
    try:
        yield
    finally:
        if previous is missing:
            # The test introduced the attribute; drop it so the next test sees
            # the same absent-attribute state this one started from.
            try:
                delattr(module.app.state, "auth_required")
            except (AttributeError, KeyError):
                # Starlette's State stores attributes in a dict and raises
                # KeyError (not AttributeError) when deleting an absent key, so
                # both are the same "already gone" no-op here.
                pass
        else:
            module.app.state.auth_required = previous
