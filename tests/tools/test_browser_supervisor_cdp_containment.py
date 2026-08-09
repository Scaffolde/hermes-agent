"""Hermetic containment tests for browser supervisor Chrome CDP fixtures."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.tools import test_browser_supervisor as supervisor_e2e


def test_devtools_active_port_is_required_for_owned_chrome_cdp(tmp_path):
    """Fixture containment uses Chrome's owned DevToolsActivePort, not a guessed port."""
    profile = tmp_path / "profile"
    profile.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        with pytest.raises(RuntimeError, match="DevToolsActivePort"):
            supervisor_e2e._read_chrome_devtools_active_port(str(profile), proc, timeout=0.01)
    finally:
        proc.wait(timeout=1)


def test_devtools_active_port_returns_owned_ephemeral_ws_path(tmp_path):
    """The owned-profile proof yields Chrome's selected port and browser path."""
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text(
        "49321\n/devtools/browser/owned-by-this-profile\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        port, browser_path = supervisor_e2e._read_chrome_devtools_active_port(
            str(profile), proc, timeout=0.5
        )
    finally:
        proc.terminate()
        proc.wait(timeout=1)

    assert port == 49321
    assert browser_path == "/devtools/browser/owned-by-this-profile"


def test_chrome_test_env_drops_cdp_override_knobs(monkeypatch):
    """A caller's browser/CDP override env cannot steer the spawned Chrome test."""
    monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9225/devtools/browser/stale")
    monkeypatch.setenv("AGENT_BROWSER_CDP_URL", "ws://127.0.0.1:9225/devtools/browser/stale")
    monkeypatch.setenv("CHROME_REMOTE_DEBUGGING_PORT", "9225")
    monkeypatch.setenv("REMOTE_DEBUGGING_PORT", "9225")
    monkeypatch.setenv("UNRELATED_BROWSER_TEST_VALUE", "kept")

    env = supervisor_e2e._chrome_test_env()

    assert "BROWSER_CDP_URL" not in env
    assert "AGENT_BROWSER_CDP_URL" not in env
    assert "CHROME_REMOTE_DEBUGGING_PORT" not in env
    assert "REMOTE_DEBUGGING_PORT" not in env
    assert env["UNRELATED_BROWSER_TEST_VALUE"] == "kept"
