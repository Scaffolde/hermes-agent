"""Tests for the gateway's periodic state.db maintenance watcher.

The construction-time auto-maintenance pass only covers process startup; a
gateway resident for weeks never pruned again, so ~/.hermes/state.db grew
without bound even with ``sessions.auto_prune`` enabled (2026-08-15
disk-pressure incident: 3GB of ~97% real rows on exactly that profile).

These tests lock the unit contract of the extracted
``_run_state_db_maintenance_once`` pass and the supervised
``_state_db_maintenance_watcher`` heartbeat that re-runs it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner


def make_runner(sessions_dir: Path) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    runner.config = MagicMock()
    runner.config.sessions_dir = sessions_dir
    session_db = MagicMock()
    session_db._db = MagicMock()
    runner._session_db = session_db
    return runner


@pytest.fixture
def sessions_cfg(monkeypatch):
    """Patch the config loader the maintenance pass re-reads on every call."""
    cfg: dict = {"sessions": {}}
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    return cfg["sessions"]


class TestMaintenanceOnce:
    def test_forwards_prune_config(self, tmp_path, sessions_cfg):
        sessions_cfg.update(
            {
                "auto_prune": True,
                "retention_days": 30,
                "min_interval_hours": 6,
                "min_vacuum_interval_days": 10,
                "vacuum_after_prune": False,
            }
        )
        runner = make_runner(tmp_path)

        runner._run_state_db_maintenance_once()

        runner._session_db._db.maybe_auto_prune_and_vacuum.assert_called_once_with(
            retention_days=30,
            min_interval_hours=6,
            min_vacuum_interval_days=10,
            vacuum=False,
            sessions_dir=tmp_path,
        )
        runner._session_db._db.maybe_auto_archive.assert_not_called()

    def test_archive_independent_of_prune(self, tmp_path, sessions_cfg):
        sessions_cfg.update({"auto_archive": True, "auto_archive_days": 5})
        runner = make_runner(tmp_path)

        runner._run_state_db_maintenance_once()

        runner._session_db._db.maybe_auto_archive.assert_called_once_with(
            idle_days=5.0,
            min_interval_hours=24,
        )
        runner._session_db._db.maybe_auto_prune_and_vacuum.assert_not_called()

    def test_disabled_config_is_a_no_op(self, tmp_path, sessions_cfg):
        runner = make_runner(tmp_path)

        runner._run_state_db_maintenance_once()

        runner._session_db._db.maybe_auto_prune_and_vacuum.assert_not_called()
        runner._session_db._db.maybe_auto_archive.assert_not_called()

    def test_no_session_db_never_raises(self, tmp_path, sessions_cfg):
        runner = make_runner(tmp_path)
        runner._session_db = None
        runner._run_state_db_maintenance_once()  # must not raise

    def test_config_loader_failure_never_raises(self, tmp_path, monkeypatch):
        import hermes_cli.config

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(hermes_cli.config, "load_config", _boom)
        runner = make_runner(tmp_path)
        runner._run_state_db_maintenance_once()  # must not raise


class TestMaintenanceWatcher:
    def test_watcher_reruns_maintenance_on_cadence(self, tmp_path, sessions_cfg, monkeypatch):
        sessions_cfg.update({"auto_prune": True})
        runner = make_runner(tmp_path)
        monkeypatch.setattr(runner, "_STATE_DB_MAINTENANCE_TICK_SECS", 0.01)

        async def _run():
            task = asyncio.create_task(runner._state_db_maintenance_watcher())
            for _ in range(200):
                await asyncio.sleep(0.01)
                if runner._session_db._db.maybe_auto_prune_and_vacuum.call_count >= 2:
                    break
            runner._running = False
            await asyncio.wait_for(task, timeout=2)

        asyncio.run(_run())
        # ≥2 calls proves the watcher re-runs on cadence, not just once.
        assert runner._session_db._db.maybe_auto_prune_and_vacuum.call_count >= 2

    def test_watcher_exits_when_stopped(self, tmp_path, sessions_cfg, monkeypatch):
        runner = make_runner(tmp_path)
        monkeypatch.setattr(runner, "_STATE_DB_MAINTENANCE_TICK_SECS", 0.01)

        async def _run():
            task = asyncio.create_task(runner._state_db_maintenance_watcher())
            await asyncio.sleep(0.03)
            runner._running = False
            await asyncio.wait_for(task, timeout=2)

        asyncio.run(_run())  # completing without timeout is the assertion
