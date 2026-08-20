"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType

import pytest


def hermes_module_names() -> list[str]:
    """Every currently-imported hermes module an isolation fixture evicts.

    The ``isinstance(name, str)`` filter is load-bearing. A test that patches
    ``importlib.util.spec_from_file_location`` with a ``MagicMock`` makes
    production code run ``sys.modules[spec.name] = module``, putting a MagicMock
    in as the KEY. ``MagicMock.startswith(...)`` auto-creates a truthy mock, so
    such a key matches every prefix below and is picked up as a hermes module —
    which would then have this package's eviction machinery applied to it.
    """
    return [
        name
        for name in list(sys.modules.keys())
        if isinstance(name, str)
        and (
            name.startswith("hermes_cli")
            or name.startswith("hermes_state")
            or name == "hermes_constants"
        )
    ]


def hermes_module_leaks(before: dict, after: dict) -> tuple[list[str], list[str]]:
    """Report hermes modules that a block dropped or swapped without restoring.

    ``before``/``after`` are ``sys.modules`` snapshots restricted to
    :func:`hermes_module_names`. Returns ``(dropped, swapped)``:

    * ``dropped`` — present before, absent after. Any spelling gets here:
      ``del sys.modules[x]``, ``sys.modules.pop(x)``, or a ``del`` through an
      alias (``m = sys.modules; del m[x]``), because this compares state, not
      source text.
    * ``swapped`` — present after, but a DIFFERENT object. This is the one that
      actually breaks later files: they hold the original identity, so a
      ``patch("hermes_cli.x.y")`` lands on the replacement while the code under
      test still calls the original (SCA-4692).

    An in-place ``importlib.reload`` mutates the existing module object rather
    than rebinding the name, so it is correctly NOT a leak. Modules imported
    for the first time during the block are additions, not leaks, and are
    likewise ignored — only the identities a later file could already be
    holding matter.
    """
    dropped = sorted(name for name in before if name not in after)
    swapped = sorted(
        name for name in before if name in after and after[name] is not before[name]
    )
    return dropped, swapped


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


# Session-wide pin: the identity each hermes module was first seen with.
#
# A per-test before/after pair is not enough. Any snapshot taken during fixture
# SETUP misses a module the test imports inside its own body, and evicting that
# module then goes unseen. That is the common case, not a corner:
# scripts/run_tests.sh gives every FILE its own interpreter, so in CI the first
# import of a hermes module usually happens inside a test body and a per-test
# snapshot starts empty. Pinning on first observation fixes the identity for the
# rest of the process the moment any test boundary sees it.
_HERMES_MODULE_IDENTITIES: dict = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Pin the identity of every hermes module already imported (SCA-4692)."""
    for name in hermes_module_names():
        _HERMES_MODULE_IDENTITIES.setdefault(name, sys.modules[name])


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """SCA-4692 class guard: fail the test that leaks a hermes module eviction.

    A behavioral invariant, not a source scan: it compares the identity of the
    hermes modules this package's isolation fixtures own (see
    :func:`hermes_module_names`) against the session pin. A hand-rolled eviction
    is caught however it is spelled — ``del``, ``.pop()``, or through an alias —
    and an eviction outside the hermes namespace (an optional dependency a test
    drops to exercise an ImportError path) is correctly ignored.

    This is a ``trylast`` hook rather than an autouse fixture on purpose. A
    fixture's teardown runs BEFORE ``monkeypatch`` undoes itself, so the guard
    saw every legitimate ``monkeypatch.setitem(sys.modules, ...)`` stub as a
    live leak — measured as 12 false errors across 6 files, on files run alone.
    ``pytest_runtest_teardown`` with ``trylast=True`` runs after all fixture
    finalization, which is the only point where "the test left global state
    dirty" is a well-defined statement.

    The originals are put back BEFORE the failure is raised. Without that, one
    offender would fail and then corrupt every file collected after it —
    precisely the 138-failure cascade with misattributed blame that SCA-4692 is
    about. Restoring first means the offender fails alone and names itself.

    The pin is only ever recorded at test setup, never from teardown state, so a
    test cannot install an impostor and have it become the baseline for later
    tests. A module first imported inside a test body is therefore unpinned, so
    ``stubbed`` covers the dangerous half of that gap: an unpinned hermes name
    left bound to something that is not a module (the usual ``MagicMock``
    stand-in) is a leak, because the next file to import it gets the stub.
    Importing a module and simply dropping it again in one body stays uncaught
    and is harmless — no earlier file holds an identity for it.
    """
    dropped, swapped = hermes_module_leaks(_HERMES_MODULE_IDENTITIES, sys.modules)
    stubbed = sorted(
        name
        for name in hermes_module_names()
        if name not in _HERMES_MODULE_IDENTITIES
        and not isinstance(sys.modules[name], ModuleType)
    )
    for name in dropped + swapped:
        sys.modules[name] = _HERMES_MODULE_IDENTITIES[name]
    for name in stubbed:
        del sys.modules[name]
    if dropped or swapped or stubbed:
        pytest.fail(
            "this test left the process-global sys.modules mutated for hermes "
            "modules; wrap the eviction in the isolated_hermes_modules / "
            "hermes_module_isolation fixture, which restores what it evicts "
            f"(SCA-4692). dropped={dropped} swapped={swapped} stubbed={stubbed}",
            pytrace=False,
        )


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

    CI never saw it because ``scripts/run_tests.sh`` gives every test FILE its
    own freshly-spawned ``python -m pytest <file>`` subprocess (no xdist, no
    shared workers — see ``.github/workflows/tests.yml`` and the run_tests.sh
    header). That mapping is 1:1 at any slice count, so no CI process ever runs
    two of these files together and the leak has nowhere to land. CI's green is
    isolation-shaped, not partition-shaped — which is why a plain
    ``pytest tests/hermes_cli`` still reports a red that CI cannot reproduce.
    Restoring here fixes the leak at its source.

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
    # `_SERVING_SESSION_TOKEN` is the once-frozen authoritative session token.
    # Freezing is correct for a real server (it stops an in-process env write
    # from invalidating live sessions) but process-global, so without a reset the
    # first test to touch the auth path would pin the token for every later test
    # — the same class of leak as `auth_required`, just one layer down. Clearing
    # it here gives each test a fresh resolution of the environment it set up.
    previous_frozen = getattr(module, "_SERVING_SESSION_TOKEN", None)
    module._SERVING_SESSION_TOKEN = None
    try:
        yield
    finally:
        module._SERVING_SESSION_TOKEN = previous_frozen
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
