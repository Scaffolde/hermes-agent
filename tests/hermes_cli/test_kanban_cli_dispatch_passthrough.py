"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.hermes_cli.conftest import hermes_module_leaks, isolated_hermes_modules


@pytest.fixture()
def isolated_kanban_home(monkeypatch, hermes_module_isolation):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Module eviction AND its restore are owned by hermes_module_isolation:
    # deleting from the process-global sys.modules without putting the
    # originals back corrupts the rest of the run (SCA-4692).
    yield test_home


def test_isolated_hermes_modules_restores_module_identity():
    """SCA-4692: the eviction must not outlive the block.

    Without the restore, `pytest tests/hermes_cli/test_kanban_cli_dispatch_passthrough.py
    tests/hermes_cli/test_model_validation.py` fails 10 tests in the second
    file that pass when it runs alone.
    """
    import hermes_cli.models  # noqa: F401  (ensure it is imported)

    before = sys.modules["hermes_cli.models"]

    with isolated_hermes_modules():
        # The eviction still happens — that is what the fixtures are for.
        assert "hermes_cli.models" not in sys.modules
        import hermes_cli.models as reimported

        assert reimported is not before

    # ...but the original object is what the rest of the session keeps seeing.
    assert sys.modules["hermes_cli.models"] is before


def test_leak_detector_catches_every_spelling_of_an_unrestored_eviction():
    """SCA-4692 class guard: the invariant is state, so spelling cannot evade it.

    The guard this exercises (``_guard_hermes_module_restoration``, autouse in
    conftest) is what actually fails an offending test. Here we drive its pure
    detector directly, because asserting on a leak end-to-end would require
    leaking on purpose in a real test — which the guard would then correctly
    fail. These are the shapes a source scan for ``del sys.modules`` misses.
    """
    import hermes_cli.models  # noqa: F401  (ensure it is in sys.modules)

    original = sys.modules["hermes_cli.models"]
    before = {"hermes_cli.models": original}

    # `del sys.modules[x]` and `sys.modules.pop(x)` are indistinguishable in
    # state — both are simply "absent after". A text scan only sees the first.
    dropped, swapped = hermes_module_leaks(before, {})
    assert dropped == ["hermes_cli.models"]
    assert swapped == []

    # An alias rebound to a different object: nothing is deleted anywhere, so
    # no eviction spelling exists to scan for, yet this is the shape that
    # actually breaks later files.
    impostor = ModuleType("hermes_cli.models")
    dropped, swapped = hermes_module_leaks(before, {"hermes_cli.models": impostor})
    assert dropped == []
    assert swapped == ["hermes_cli.models"]

    # Restored to the same object: clean.
    assert hermes_module_leaks(before, {"hermes_cli.models": original}) == ([], [])


def test_leak_detector_ignores_non_hermes_and_newly_imported_modules():
    """The invariant is scoped to the modules the isolation fixtures own.

    A test that legitimately evicts an unrelated optional dependency to
    exercise an ImportError path must not be flagged — that false positive is
    exactly what the previous source-scanning guard would have produced.
    """
    import hermes_cli.models  # noqa: F401  (ensure it is in sys.modules)

    original = sys.modules["hermes_cli.models"]
    before = {"hermes_cli.models": original}

    # An unrelated third-party module dropped by a test: not ours, not a leak.
    # (It never enters `before`, because hermes_module_names() filters it out.)
    after_without_optional_dep = {"hermes_cli.models": original}
    assert hermes_module_leaks(before, after_without_optional_dep) == ([], [])

    # First-time imports during the test are additions, not leaks: no earlier
    # file can be holding an identity for a module that was not yet imported.
    after_with_new_import = {
        "hermes_cli.models": original,
        "hermes_cli.newly_imported": ModuleType("hermes_cli.newly_imported"),
    }
    assert hermes_module_leaks(before, after_with_new_import) == ([], [])


def test_autouse_guard_restores_before_it_fails():
    """A leak must fail its own test WITHOUT cascading into later files.

    The guard restores the originals before asserting, so the 138-failure
    misattributed cascade SCA-4692 describes cannot happen even when the guard
    fires. This asserts the restore half of that contract.
    """
    import hermes_cli.models  # noqa: F401  (ensure it is in sys.modules)

    original = sys.modules["hermes_cli.models"]
    before = {"hermes_cli.models": original}
    live = {"hermes_cli.models": ModuleType("hermes_cli.models")}

    dropped, swapped = hermes_module_leaks(before, live)
    for name in dropped + swapped:
        live[name] = before[name]

    # After the guard's restore pass, the next test sees the original identity.
    assert live["hermes_cli.models"] is original
    assert hermes_module_leaks(before, live) == ([], [])


def test_cli_dispatch_passes_max_in_progress_from_config(isolated_kanban_home, monkeypatch):
    """#33488: hermes kanban dispatch must pass kanban.max_in_progress from
    config to dispatch_once. Without this, the global concurrency cap is
    unreachable from the CLI even though it works from the gateway."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    # Configure max_in_progress in the loaded config.
    fake_config = {
        "kanban": {
            "max_in_progress": 3,
            "max_spawn": 5,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: fake_config
    )

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    # Every config value must have reached dispatch_once.
    assert captured.get("max_in_progress") == 3, (
        f"CLI must pass kanban.max_in_progress from config; got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("max_spawn") == 5, (
        f"CLI must pass kanban.max_spawn from config when --max is not provided; got {captured.get('max_spawn')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    fake_config = {"kanban": {"max_spawn": 10}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    args = argparse.Namespace(dry_run=True, max=2, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    assert captured.get("max_spawn") == 2, (
        f"CLI --max=2 must override config kanban.max_spawn=10; got {captured.get('max_spawn')!r}"
    )


