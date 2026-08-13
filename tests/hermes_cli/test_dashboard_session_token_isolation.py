"""Regression pins for the two SCA-4692 test-isolation root causes.

Both defects were invisible to CI because the dashboard-auth files are pinned to
a single xdist worker: the leaks stayed inside that worker, so the suite was
green as a property of the sharding rather than of the code. Running
``tests/hermes_cli`` as one process failed 138+ tests across 57 files.
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
    """

    def test_env_set_after_import_is_honoured(self, monkeypatch):
        # web_server is imported at module scope above, i.e. strictly BEFORE
        # this env var is set — the exact ordering that used to break.
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "late-bound-token")
        assert web_server._session_token() == "late-bound-token"

    def test_env_change_is_picked_up_without_reimport(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "first-token")
        assert web_server._session_token() == "first-token"
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "second-token")
        assert web_server._session_token() == "second-token"

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
