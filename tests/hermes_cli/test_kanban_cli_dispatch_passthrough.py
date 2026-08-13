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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.hermes_cli.conftest import isolated_hermes_modules


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


def test_no_unrestored_sys_modules_eviction_in_hermes_cli_tests():
    """SCA-4692 class guard: no test file may evict hermes modules by hand.

    The one legal mechanism is conftest's ``isolated_hermes_modules`` /
    ``hermes_module_isolation``, which restores what it evicted. A bare
    hand-rolled eviction in a fixture leaks process-global state into every
    test file collected after it.
    """
    # Assembled at runtime so this guard does not match its own source.
    needle = "del sys." + "modules"
    tests_dir = Path(__file__).parent
    offenders = sorted(
        path.name
        for path in tests_dir.glob("test_*.py")
        if needle in path.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        "these hermes_cli test files evict hermes modules by hand; use the "
        f"hermes_module_isolation fixture instead: {offenders}"
    )


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


