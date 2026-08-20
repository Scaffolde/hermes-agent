"""Tests for the structured logging dedup model.

The contract: a 1000-write session in one project should emit exactly
ONE INFO line ("active for <root>") at the default INFO threshold.
Steady-state events stay at DEBUG; first-time-seen events surface
once at INFO/WARNING.
"""
from __future__ import annotations

import logging

import pytest

from agent.lsp import eventlog


@pytest.fixture(autouse=True)
def _reset():
    eventlog.reset_announce_caches()
    yield
    eventlog.reset_announce_caches()


@pytest.fixture
def caplog_lsp(caplog):
    """Capture `hermes.lint.lsp` records WITHOUT letting them escape.

    `caplog.set_level()` leaves propagation on, so records still climb to the
    root logger. If a production file handler is installed in this process they
    land in the operator's real `agent.log` — and at DEBUG, so even the
    steady-state events the eventlog design deliberately keeps below INFO get
    written out. Disabling propagation keeps them in caplog's handler, which is
    all these tests read.

    Capture has to survive that. pytest's capture handler lives on the *root*
    logger, so switching propagation off could cut these tests off from their
    own records. Under the pinned pytest, `set_level(logger=...)` also binds the
    handler to the named logger, so it does not — but that is an implementation
    detail rather than a documented promise. The bind below makes the capture
    path explicit and self-contained: it is a no-op today (pytest has already
    attached the same handler object), and becomes load-bearing only if that
    behaviour ever changes. Teardown removes only a handler this fixture added,
    so pytest's own handler lifecycle is left alone.
    """
    lg = logging.getLogger("hermes.lint.lsp")
    previous = lg.propagate
    caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
    attached_here = caplog.handler not in lg.handlers
    if attached_here:
        lg.addHandler(caplog.handler)
    lg.propagate = False
    try:
        yield caplog
    finally:
        lg.propagate = previous
        if attached_here:
            lg.removeHandler(caplog.handler)


# ---------------------------------------------------------------------------
# Steady-state silence (DEBUG)
# ---------------------------------------------------------------------------


def test_clean_emits_at_debug(caplog_lsp):
    for _ in range(10):
        eventlog.log_clean("pyright", "/proj/x.py")
    info_records = [r for r in caplog_lsp.records if r.levelno >= logging.INFO]
    debug_records = [r for r in caplog_lsp.records if r.levelno == logging.DEBUG]
    assert info_records == []
    assert len(debug_records) == 10


def test_disabled_emits_at_debug(caplog_lsp):
    eventlog.log_disabled("pyright", "/x.py", "feature off")
    eventlog.log_disabled("pyright", "/x.py", "ext not mapped")
    assert all(r.levelno == logging.DEBUG for r in caplog_lsp.records)


# ---------------------------------------------------------------------------
# State transitions: INFO once, DEBUG thereafter
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Diagnostics events fire INFO every time
# ---------------------------------------------------------------------------


def test_diagnostics_always_info(caplog_lsp):
    for i in range(5):
        eventlog.log_diagnostics("pyright", f"/x{i}.py", 1)
    info = [r for r in caplog_lsp.records if r.levelno == logging.INFO]
    assert len(info) == 5
    assert all("diags" in r.getMessage() for r in info)


# ---------------------------------------------------------------------------
# Action-required: WARNING once, DEBUG thereafter (or per call for novel events)
# ---------------------------------------------------------------------------












def test_spawn_failed_warns(caplog_lsp):
    eventlog.log_spawn_failed("pyright", "/proj", FileNotFoundError("nope"))
    warns = [r for r in caplog_lsp.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "spawn/initialize failed" in warns[0].getMessage()


# ---------------------------------------------------------------------------
# Format: log lines all carry the lsp[<server_id>] prefix for grep
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Steady-state contract: 1000 clean writes → 1 INFO at most
# ---------------------------------------------------------------------------


def test_thousand_clean_writes_emit_one_info(caplog_lsp):
    """A long session writes lots of files cleanly; agent.log should
    show ONE 'active for' INFO and zero other INFO lines."""
    eventlog.log_active("pyright", "/proj")
    for _ in range(1000):
        eventlog.log_clean("pyright", "/proj/x.py")
    info_records = [r for r in caplog_lsp.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert "active for" in info_records[0].getMessage()


# ---------------------------------------------------------------------------
# Containment: these records must never reach the root logger's handlers
# ---------------------------------------------------------------------------


def test_records_do_not_escape_to_root_handlers(caplog_lsp):
    """The fixture must keep records out of every handler but caplog's.

    Without this, `caplog.set_level()` leaves propagation on and the DEBUG
    records climb to root. In any process holding a production file handler
    that means writes into the operator's `~/.hermes/logs/agent.log`, where
    the synthetic `/x*.py` paths below are indistinguishable from real
    language-server traffic to anyone replaying the log.
    """
    escaped: list[logging.LogRecord] = []

    class _Spy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name == "hermes.lint.lsp":
                escaped.append(record)

    spy = _Spy(level=logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(spy)
    previous_root_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        eventlog.log_active("pyright", "/proj")
        eventlog.log_clean("pyright", "/proj/x.py")
        for i in range(5):
            eventlog.log_diagnostics("pyright", f"/x{i}.py", 1)
        eventlog.log_spawn_failed("pyright", "/proj", FileNotFoundError("nope"))
    finally:
        root.removeHandler(spy)
        root.setLevel(previous_root_level)

    # caplog still sees everything — containment must not cost coverage.
    assert caplog_lsp.records, "fixture stopped capturing records"
    assert escaped == [], (
        f"{len(escaped)} lsp record(s) reached the root logger and would be "
        "written to any production file handler installed in this process"
    )


# ---------------------------------------------------------------------------
# Path shortening
# ---------------------------------------------------------------------------




def test_short_path_keeps_absolute_when_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "a") if (tmp_path / "a").exists() else None
    monkeypatch.chdir(tmp_path)
    other = "/var/log/foo.txt"
    out = eventlog._short_path(other)
    # Outside cwd: keeps absolute (no leading "../")
    assert out == "/var/log/foo.txt" or not out.startswith("..")


