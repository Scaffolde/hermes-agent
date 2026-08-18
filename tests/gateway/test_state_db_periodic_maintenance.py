"""Tests for periodic state.db retention on long-lived gateways.

The sessions auto-maintenance sweep (archive + prune ended sessions +
throttled VACUUM) only ever ran at process construction; a gateway resident
for weeks never pruned again, so ~/.hermes/state.db grew without bound even
with ``sessions.auto_prune`` enabled (2026-08-15 disk-pressure incident:
3GB of ~97% real rows on exactly that profile).

Two surfaces under test:

* ``GatewayRunner._run_state_db_maintenance_once`` — the extracted
  construction-time pass (the only pass allowed to VACUUM, since it runs
  before traffic is served).
* ``_start_gateway_housekeeping``'s hourly session-maintenance branch —
  the recurring cadence: archive + prune per served profile, always with
  ``vacuum=False`` so a mid-traffic exclusive-lock rewrite can never starve
  active turns' persistence, and never two schedulers for the same chore.
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.run as gw_run
from gateway.run import GatewayRunner, _start_gateway_housekeeping


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
    """Patch the config loader the maintenance passes re-read on every call."""
    cfg: dict = {"sessions": {}}
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    return cfg["sessions"]


class TestStartupMaintenancePass:
    def test_forwards_prune_config_with_vacuum(self, tmp_path, sessions_cfg):
        sessions_cfg.update(
            {
                "auto_prune": True,
                "retention_days": 30,
                "min_interval_hours": 6,
                "min_vacuum_interval_days": 10,
                "vacuum_after_prune": True,
            }
        )
        runner = make_runner(tmp_path)

        runner._run_state_db_maintenance_once()

        # Startup is the one pass where vacuum=True is allowed (pre-traffic).
        runner._session_db._db.maybe_auto_prune_and_vacuum.assert_called_once_with(
            retention_days=30,
            min_interval_hours=6,
            min_vacuum_interval_days=10,
            vacuum=True,
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


def run_housekeeping_session_tick(
    monkeypatch,
    tmp_path,
    *,
    sessions_cfg: dict,
    profiles: list[tuple[str, Path]],
    multiplex: bool,
) -> list[MagicMock]:
    """Drive _start_gateway_housekeeping through exactly one hourly tick.

    Media/paste/curator/memory chores are neutralized; profiles_to_serve and
    the profile scope are patched so the test controls the served set. One
    SessionDB mock is minted per SessionDB() call and returned for
    assertions.
    """
    import hermes_cli.config
    import hermes_cli.profiles
    import hermes_state

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"sessions": dict(sessions_cfg)})

    seen_multiplex: list[bool] = []

    def fake_profiles_to_serve(multiplex: bool):
        seen_multiplex.append(multiplex)
        return list(profiles)

    monkeypatch.setattr(hermes_cli.profiles, "profiles_to_serve", fake_profiles_to_serve)

    scoped_homes: list[Path] = []

    from contextlib import contextmanager

    @contextmanager
    def fake_scope(profile_home):
        scoped_homes.append(Path(profile_home))
        yield

    monkeypatch.setattr(gw_run, "_profile_runtime_scope", fake_scope)

    minted: list[MagicMock] = []

    def fake_session_db():
        db = MagicMock()
        minted.append(db)
        return db

    monkeypatch.setattr(hermes_state, "SessionDB", fake_session_db)

    # Neutralize the other hourly chores so the tick is deterministic.
    for name in (
        "cleanup_audio_cache",
        "cleanup_document_cache",
        "cleanup_image_cache",
        "cleanup_screenshot_cache",
        "cleanup_video_cache",
    ):
        import gateway.platforms.base as plat_base

        monkeypatch.setattr(plat_base, name, lambda max_age_hours=24: 0)
    import hermes_cli.debug as dbg

    monkeypatch.setattr(dbg, "_sweep_expired_pastes", lambda: (0, 0))
    import agent.curator as curator_mod

    monkeypatch.setattr(curator_mod, "maybe_run_curator", lambda **kw: None)
    import tools.skills_sync_client as sync_mod

    monkeypatch.setattr(sync_mod, "maybe_pull_skills", lambda: None)
    monkeypatch.setattr(sync_mod, "maybe_pull_org_skills", lambda: None)
    import hermes_cli.mem_trim as mem_trim_mod

    monkeypatch.setattr(mem_trim_mod, "trim_memory", lambda reason=None: None)

    # Stop after the 60th tick (the hourly branch) has run: the stop_event is
    # set from inside a fake time-waiter to avoid a wall-clock hour.
    stop_event = threading.Event()
    ticks = {"n": 0}

    original_wait = stop_event.wait

    def counting_wait(timeout=None):
        ticks["n"] += 1
        if ticks["n"] >= 60:
            stop_event.set()
        return original_wait(timeout=0)

    stop_event.wait = counting_wait  # type: ignore[method-assign]

    _start_gateway_housekeeping(stop_event, adapters=None, loop=None, interval=0, multiplex=multiplex)

    assert seen_multiplex == [multiplex]
    assert scoped_homes == [Path(h) for _, h in profiles]
    return minted


class TestHousekeepingSessionMaintenance:
    def test_prunes_without_vacuum_on_live_tick(self, monkeypatch, tmp_path):
        home = tmp_path / "home"
        minted = run_housekeeping_session_tick(
            monkeypatch,
            tmp_path,
            sessions_cfg={"auto_prune": True, "retention_days": 45},
            profiles=[("default", home)],
            multiplex=False,
        )
        assert len(minted) == 1
        db = minted[0]
        db.maybe_auto_prune_and_vacuum.assert_called_once_with(
            retention_days=45,
            min_interval_hours=24,
            min_vacuum_interval_days=30,
            # Live ticks must NEVER vacuum: exclusive full-rewrite lock while
            # the gateway is serving traffic.
            vacuum=False,
            sessions_dir=home / "sessions",
        )
        db.maybe_auto_archive.assert_not_called()
        db.close.assert_called_once()

    def test_multiplex_sweeps_every_profile(self, monkeypatch, tmp_path):
        homes = [
            ("default", tmp_path / "default"),
            ("work", tmp_path / "profiles" / "work"),
            ("personal", tmp_path / "profiles" / "personal"),
        ]
        minted = run_housekeeping_session_tick(
            monkeypatch,
            tmp_path,
            sessions_cfg={"auto_prune": True, "auto_archive": True},
            profiles=homes,
            multiplex=True,
        )
        assert len(minted) == 3
        for (_, home), db in zip(homes, minted):
            db.maybe_auto_prune_and_vacuum.assert_called_once()
            assert db.maybe_auto_prune_and_vacuum.call_args.kwargs["vacuum"] is False
            assert (
                db.maybe_auto_prune_and_vacuum.call_args.kwargs["sessions_dir"]
                == Path(home) / "sessions"
            )
            db.maybe_auto_archive.assert_called_once()
            db.close.assert_called_once()

    def test_disabled_profiles_open_no_db(self, monkeypatch, tmp_path):
        minted = run_housekeeping_session_tick(
            monkeypatch,
            tmp_path,
            sessions_cfg={},
            profiles=[("default", tmp_path / "home")],
            multiplex=False,
        )
        assert minted == []
