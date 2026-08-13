"""Regression pins for the two SCA-4692 test-isolation root causes.

Both defects were invisible to CI because ``scripts/run_tests.sh`` runs every
test FILE in its own freshly-spawned ``python -m pytest <file>`` subprocess — no
xdist, no shared workers. That file-to-process mapping is 1:1 at any slice count,
so no CI process ever runs two of these files together and the leaks have nowhere
to land. CI's green is isolation-shaped, not partition-shaped, which is exactly
why a plain ``pytest tests/hermes_cli`` failed 138+ tests across 57 files with a
red CI could never reproduce.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from hermes_cli import web_server  # noqa: E402


class TestSessionTokenResolvedAtCallTime:
    """`_SESSION_TOKEN` used to be bound at MODULE IMPORT time.

    That made the effective token depend on which module imported web_server
    first: anything exporting HERMES_DASHBOARD_SESSION_TOKEN afterwards got a
    server that had already pinned a different token, and every request 401'd.

    The resolution is now FROZEN once at the serve boundary rather than re-read
    on every check — late enough that import order cannot pin it, and exactly
    once, so an in-process env write cannot invalidate live sessions. The two
    halves are pinned here and in :class:`TestFrozenTokenSurvivesEnvRotation`.
    """

    def test_env_set_after_import_is_honoured(self, monkeypatch):
        # web_server is imported at module scope above, i.e. strictly BEFORE
        # this env var is set — the exact ordering that used to break.
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "late-bound-token")
        assert web_server._session_token() == "late-bound-token"

    def test_freeze_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "first-token")
        assert web_server._session_token() == "first-token"
        # Repeated resolution must be stable: the value is injected into the SPA
        # HTML and compared against on every later request.
        assert web_server._session_token() == "first-token"
        assert web_server._freeze_session_token() == "first-token"

    def test_falls_back_to_stable_process_token(self, monkeypatch):
        # No env override: the fallback must be STABLE across calls, because it
        # is injected into the SPA HTML and compared against on later requests.
        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        web_server._SSH_SESSION_TOKEN = None
        assert web_server._session_token() == web_server._session_token()
        assert web_server._session_token() == web_server._SESSION_TOKEN

    def test_ssh_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "env-token")
        previous = web_server._SSH_SESSION_TOKEN
        previous_fallback = web_server._SESSION_TOKEN
        try:
            web_server._apply_ssh_session_token("s" * 64)
            assert web_server._session_token() == "s" * 64
            # Kept in step so by-value readers (web_deps.get_session_token,
            # tests importing the attribute) observe the SSH token too.
            assert web_server._SESSION_TOKEN == "s" * 64
        finally:
            web_server._SSH_SESSION_TOKEN = previous
            web_server._SESSION_TOKEN = previous_fallback


class TestFrozenTokenSurvivesEnvRotation:
    """A live session must not be 401'd by an in-process env write.

    `PUT /api/env` → `save_provider_env_credential()` → `save_env_value()`
    assigns straight into `os.environ`, and HERMES_DASHBOARD_SESSION_TOKEN is
    NOT on the writer denylist (that list covers PATH / PYTHONPATH / HERMES_HOME
    and friends). So a loopback-dashboard user who writes or rotates that key
    through the authenticated env editor mutates this process's environment.

    If the authoritative token were re-read on every auth check, that write would
    swap the credential mid-flight while the browser kept sending the token
    injected into the SPA HTML — every later REST and WebSocket request from an
    already-open session would 401. Freezing at the serve boundary is what
    prevents it.
    """

    @pytest.fixture
    def hermes_home(self, monkeypatch, tmp_path):
        home = tmp_path / "session_token_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        from hermes_cli.config import invalidate_env_cache

        invalidate_env_cache()
        return home

    def test_rotating_the_token_through_the_env_api_keeps_live_sessions_valid(
        self, hermes_home, monkeypatch
    ):
        import os

        from fastapi.testclient import TestClient

        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        monkeypatch.setattr(web_server, "_SSH_SESSION_TOKEN", None)

        client = TestClient(web_server.app)

        # Freeze exactly as the serve boundary would, then hand that token to a
        # client — this stands in for the value injected into the SPA HTML.
        serving_token = web_server._freeze_session_token()
        headers = {web_server._SESSION_HEADER_NAME: serving_token}

        rotated = "rotated-" + "z" * 32
        resp = client.put(
            "/api/env",
            json={"key": "HERMES_DASHBOARD_SESSION_TOKEN", "value": rotated},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text

        # Positive control on the hazard's mechanism: the write really did land
        # in THIS process's environment. Without this assertion the test could
        # pass simply because nothing was written.
        assert os.environ["HERMES_DASHBOARD_SESSION_TOKEN"] == rotated

        # The property: the authoritative token is unchanged, so the token the
        # client already holds still authenticates.
        assert web_server._session_token() == serving_token
        assert web_server._session_token() != rotated

        follow_up = client.get("/api/env", headers=headers)
        assert follow_up.status_code == 200, follow_up.text


class TestAuthRequiredDoesNotLeak:
    """`app` is a module-level singleton, so `app.state` is process-global.

    The gate tests left `auth_required` True and every later test that reused
    `app` hit the gated branch and got 401. These two run in file order: the
    first dirties the flag, the second proves the autouse fixture in
    ``tests/hermes_cli/conftest.py`` put it back.
    """

    def test_a_dirties_auth_required(self):
        web_server.app.state.auth_required = True
        assert web_server.app.state.auth_required is True

    def test_b_sees_a_clean_flag(self):
        assert getattr(web_server.app.state, "auth_required", False) is not True
