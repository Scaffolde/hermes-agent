"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import contextlib
import sys

import pytest


def hermes_module_names() -> list[str]:
    """Every currently-imported hermes module an isolation fixture evicts."""
    return [
        name
        for name in list(sys.modules.keys())
        if name.startswith("hermes_cli")
        or name.startswith("hermes_state")
        or name == "hermes_constants"
    ]


@contextlib.contextmanager
def isolated_hermes_modules():
    """Evict the hermes modules for the duration of the block, then restore.

    The eviction forces the hermes modules to re-import against the caller's
    temporary ``HERMES_HOME`` instead of whatever the process already had
    bound. It MUST be undone: ``sys.modules`` is process-global, and every
    other test module in the run captured its imports at collection time.

    Leaving the eviction in place means a later ``patch("hermes_cli.x.y")``
    re-imports a SECOND module object and patches that, while the test under
    test still calls the function bound to the original object — so the patch
    silently does nothing and unrelated tests fail for a cause they never
    triggered (SCA-4692).
    """
    saved = {name: sys.modules[name] for name in hermes_module_names()}
    for name in saved:
        del sys.modules[name]
    try:
        yield saved
    finally:
        # Drop whatever the block imported against the temp HERMES_HOME,
        # then put the original module objects back so identities that other
        # test modules already hold stay valid.
        for name in hermes_module_names():
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture()
def hermes_module_isolation():
    """Fixture form of :func:`isolated_hermes_modules`.

    Request it from any fixture that needs a fresh ``HERMES_HOME`` to be
    re-imported. Eviction happens during this fixture's setup, so a
    requesting fixture's own body (which sets ``HERMES_HOME`` and then
    imports) still sees a cleared module cache; the restore runs after the
    requesting fixture tears down.
    """
    with isolated_hermes_modules() as saved:
        yield saved


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
