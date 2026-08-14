"""Tests for the two LSP bounds an operator can actually configure.

``max_clients`` bounds how *many* language servers exist; ``idle_timeout``
bounds how *long* each one lives.  Only the first was defended.
:func:`agent.lsp.manager.resolve_max_clients` rejects zero, negatives,
strings and the non-finite floats, but ``idle_timeout`` was coerced
inline with a bare ``float()`` and a ``0 < x < MIN`` clamp — and ``.nan``
/ ``.inf`` are valid YAML that survive ``float()`` without raising and
slip straight through that clamp.  Driving the real factory showed both
halves of the resulting hole:

    idle_timeout=nan -> _idle_timeout=nan  starts=False cutoff= nan
    idle_timeout=inf -> _idle_timeout=inf  starts=True  cutoff=-inf

At ``nan`` the reaper never starts (``nan > 0`` is False); at ``inf`` it
starts and reaps nothing, because ``_reap_idle_once`` selects on
``_last_used < cutoff`` and no stamp is below ``-inf``.  The count cap
still binds either way, so this is not unbounded accumulation — it is
the reaper's own axis silently switched off, leaving the fleet pinned at
``max_clients`` forever.

The second half of this file covers ``get_status()``.  ``hermes lsp
status`` is how an operator audits the bounds without reading source; it
reported neither the limits in force nor per-client ages, which is
exactly the visibility that made SCA-4621 and SCA-4583 diagnosable.
"""
from __future__ import annotations

import logging
from typing import Optional

import pytest

from agent.lsp import manager as manager_mod
from agent.lsp.manager import (
    DEFAULT_IDLE_TIMEOUT,
    MIN_IDLE_TIMEOUT,
    LSPService,
    default_max_clients,
)


class FakeClient:
    """Stands in for ``LSPClient``: identity, state, awaitable shutdown."""

    def __init__(self, server_id: str, workspace_root: str) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.state = "ready"
        self.is_running = True
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def make_service(
    idle_timeout: float = 600.0,
    max_clients: Optional[int] = None,
) -> LSPService:
    """A service with no background loop and no real subprocesses.

    ``memory_budget=None`` disables the byte half of the cap so these
    stay host-independent, matching ``test_client_cap.py``.
    """
    return LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=idle_timeout,
        max_clients=max_clients,
        memory_budget=None,
    )


def inject(svc: LSPService, root: str, idle_for: float = 0.0) -> FakeClient:
    """Install a fake client that was last used *idle_for* seconds ago.

    The stamp is taken from the service's OWN clock via ``_touch`` and
    then aged, never written as a raw ``time.time()``.  The reaper and
    ``get_status`` must read the same clock as ``_last_used``; deriving
    the fixture's stamp from that clock is what lets
    :func:`test_status_idle_seconds_reads_the_same_clock_as_last_used`
    fail loudly if the two ever diverge, instead of silently reporting
    an epoch difference as an age.
    """
    key = ("pyright", root)
    client = FakeClient("pyright", root)
    svc._clients[key] = client
    svc._touch(client)
    svc._last_used[key] -= idle_for
    return client


# ----------------------------------------------------------------------
# idle_timeout coercion — the guard max_clients already has
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (42, 42.0),
        ("42", 42.0),
        (0, 0.0),                       # documented: disables reaping
        ("0", 0.0),
        (5, MIN_IDLE_TIMEOUT),          # below the floor, clamped
        (None, DEFAULT_IDLE_TIMEOUT),
        (-5, DEFAULT_IDLE_TIMEOUT),     # a silent disable, not a request
        ("nonsense", DEFAULT_IDLE_TIMEOUT),
        ({}, DEFAULT_IDLE_TIMEOUT),
        (float("nan"), DEFAULT_IDLE_TIMEOUT),
        (float("inf"), DEFAULT_IDLE_TIMEOUT),
        (float("-inf"), DEFAULT_IDLE_TIMEOUT),
        (False, DEFAULT_IDLE_TIMEOUT),  # YAML 1.1 `off`/`no`, not a 0
        (True, DEFAULT_IDLE_TIMEOUT),   # YAML 1.1 `on`/`yes`, not a 1
    ],
)
def test_resolve_idle_timeout_matches_the_max_clients_treatment(raw, expected):
    """Both knobs coerce through a named resolver, so they cannot drift.

    ``0`` is the one non-positive value that survives: the user guide
    documents it as "hold every server's index warm for the life of the
    process".  A negative does not mean that to anyone, and it reaches
    the same place by accident.
    """
    assert manager_mod.resolve_idle_timeout(raw) == expected


def test_boolean_bounds_fall_back_instead_of_coercing_to_a_number():
    """``idle_timeout: off`` is a bool, and ``float(False)`` is a silent 0.

    YAML 1.1 resolves ``off``/``no`` to ``False`` and ``on``/``yes`` to
    ``True``, so an operator who writes ``idle_timeout: off`` meaning
    "no timeout" hands the resolver a ``bool``.  ``bool`` is a subclass
    of ``int``, so it never raises on the way through ``float()``:

        idle_timeout: off -> 0.0  -> reads as the documented "0", which
                                     disables the reaper outright
        idle_timeout: on  -> 1.0  -> clamped up to MIN_IDLE_TIMEOUT
        max_clients:  on  -> 1    -> a one-server fleet, silently

    ``off`` reaching ``0.0`` is the same disarmed reaper this whole
    module exists to prevent, arrived at by a different door — and none
    of the three warn, because ``_bound_was_overridden`` compares
    numerically and ``float(False) == 0.0`` is the value it resolved to.
    A bool is not a number an operator can have meant; it derives the
    default, matching the ``isinstance(raw, bool)`` rejection already
    used in ``agent/retry_utils.py`` and ``agent/image_routing.py``.
    """
    for raw in (False, True):
        assert manager_mod.resolve_idle_timeout(raw) == DEFAULT_IDLE_TIMEOUT, raw
        assert manager_mod.resolve_max_clients(raw) == default_max_clients(), raw


def test_a_boolean_bound_is_logged_as_an_override(monkeypatch, caplog):
    """The fallback is loud: a bool must not disarm the reaper in silence.

    ``_bound_was_overridden`` decides whether the operator hears about a
    rejected value, and it compares numerically — so ``False`` against a
    resolved ``600.0`` must still read as an override rather than being
    coerced back to ``0.0`` for the comparison and passing as unchanged.
    """
    cfg = {"lsp": {"enabled": False, "max_clients": True, "idle_timeout": False}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    with caplog.at_level(logging.WARNING):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._max_clients == default_max_clients()
        assert "idle_timeout" in caplog.text
        assert "max_clients" in caplog.text
    finally:
        svc.shutdown()


def test_non_finite_bounds_fall_back_instead_of_crashing_the_lsp_path(monkeypatch):
    """``idle_timeout: .nan`` is valid YAML and must not disarm the reaper.

    Ported from #50.  ``float('.nan')`` survives the parse and slips past
    the ``0 < x < MIN`` clamp; ``nan`` then fails the ``> 0`` start guard
    so the reaper never starts, and ``inf`` starts a reaper whose cutoff
    is ``-inf`` and which therefore reaps nothing.  ``max_clients`` has
    been covered since ``resolve_max_clients`` landed; this asserts both
    halves so a future edit cannot fix one and forget the other.
    """
    for bad in (".nan", ".inf", "-.inf", float("nan"), float("inf")):
        cfg = {"lsp": {"enabled": False, "max_clients": bad, "idle_timeout": bad}}
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda cfg=cfg: cfg
        )
        svc = LSPService.create_from_config()
        assert svc is not None, f"{bad!r} took down create_from_config()"
        try:
            assert svc._max_clients == default_max_clients(), bad
            assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT, bad
        finally:
            svc.shutdown()


def test_create_from_config_honours_usable_bounds(monkeypatch):
    """The positive control: a real config still reaches the service.

    Without this, every assertion above passes just as happily against a
    resolver hard-wired to return the defaults.
    """
    cfg = {"lsp": {"enabled": False, "idle_timeout": 42, "max_clients": 3}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == 42.0
        assert svc._max_clients == 3
    finally:
        svc.shutdown()


def test_config_idle_timeout_zero_still_disables_reaping(monkeypatch):
    """``idle_timeout: 0`` is documented and must survive the new guard.

    Coercing it to the default would quietly start reaping on a host
    whose operator asked to keep every index warm.
    """
    cfg = {"lsp": {"enabled": False, "idle_timeout": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == 0.0
    finally:
        svc.shutdown()


def test_config_max_clients_zero_is_rejected_loudly(monkeypatch, caplog):
    """A non-positive cap falls back to the host default — and says so.

    Ported from #50.  ``max_clients: 0`` reads like "no limit", but the
    cap is deliberately not disableable: an unbounded population put the
    host into swap and took self-hosted CI offline.  Falling back is
    right; doing it at ``debug`` is not, because the operator then
    believes a bound they asked to remove is gone when it is still on.
    """
    cfg = {"lsp": {"enabled": False, "max_clients": 0}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    with caplog.at_level(logging.WARNING, logger="agent.lsp.manager"):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._max_clients == default_max_clients()
    finally:
        svc.shutdown()

    assert any(
        "max_clients" in r.getMessage() for r in caplog.records
    ), "a rejected cap must be logged at WARNING, not swallowed at debug"


def test_config_idle_timeout_garbage_is_rejected_loudly(monkeypatch, caplog):
    """Same contract for the other knob: overridden means visible."""
    cfg = {"lsp": {"enabled": False, "idle_timeout": ".nan"}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    with caplog.at_level(logging.WARNING, logger="agent.lsp.manager"):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
    finally:
        svc.shutdown()

    assert any(
        "idle_timeout" in r.getMessage() for r in caplog.records
    ), "a rejected idle timeout must be logged at WARNING"


def test_a_usable_bound_is_not_logged_as_an_override(monkeypatch, caplog):
    """The negative control for the two tests above.

    A warning on every startup is a warning nobody reads, so the loud
    path must fire only when a value was actually replaced.
    """
    cfg = {"lsp": {"enabled": False, "idle_timeout": 600, "max_clients": 4}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

    with caplog.at_level(logging.WARNING, logger="agent.lsp.manager"):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        assert svc._idle_timeout == 600.0
        assert svc._max_clients == 4
    finally:
        svc.shutdown()

    assert not [
        r for r in caplog.records
        if "idle_timeout" in r.getMessage() or "max_clients" in r.getMessage()
    ], "an accepted bound must not warn"


# ----------------------------------------------------------------------
# get_status — the bounds an operator can see
# ----------------------------------------------------------------------


def test_status_reports_the_bounds():
    """``hermes lsp status`` should show the limits actually in force.

    Ported from #50.  Without these an operator auditing a host that is
    pinned at its cap has no way to see what the cap *is* short of
    reading ``manager.py`` and re-deriving it from host RAM.
    """
    svc = make_service(idle_timeout=123.0, max_clients=7)
    try:
        inject(svc, "/tmp/ws-status", idle_for=5.0)
        status = svc.get_status()

        assert status["idle_timeout"] == 123.0
        assert status["max_clients"] == 7
        client = status["clients"][0]
        assert client["idle_seconds"] >= 5.0
        assert client["inflight"] == 0
    finally:
        svc.shutdown()


def test_status_idle_seconds_reads_the_same_clock_as_last_used():
    """``idle_seconds`` must be an age, not a difference of two epochs.

    ``_last_used`` and the status age are only ever compared to each
    other.  If ``get_status`` reads a different clock than the one
    writing ``_last_used`` — the exact hazard when the reaper's clock
    moves, as in #62 — every age becomes the gap between the two epochs
    (order 1e9 for wall-clock-minus-monotonic) and still satisfies a
    bare ``>= 5``.  Bounding it on both sides is what makes that fail.
    """
    svc = make_service(idle_timeout=600.0)
    try:
        inject(svc, "/tmp/ws-clock", idle_for=5.0)
        idle = svc.get_status()["clients"][0]["idle_seconds"]

        assert 5.0 <= idle < 60.0, (
            f"idle_seconds={idle!r} is not an elapsed time — get_status and "
            "_last_used are reading different clocks"
        )
    finally:
        svc.shutdown()


def test_status_counts_inflight_requests():
    """``inflight`` is what explains a client the cap refuses to evict."""
    svc = make_service(idle_timeout=600.0, max_clients=4)
    try:
        inject(svc, "/tmp/ws-busy")
        svc._inflight[("pyright", "/tmp/ws-busy")] = 2

        assert svc.get_status()["clients"][0]["inflight"] == 2
    finally:
        svc.shutdown()


def test_human_readable_status_renders_the_bounds(monkeypatch, capsys):
    """``hermes lsp status`` (no --json) must show the bounds and ages.

    Ported from #50, and driven through the real ``_cmd_status`` rather
    than a re-implementation of its format string: the operator
    auditability this feature claims is worthless if only the
    machine-readable path carries it, and a test that renders its own
    copy of the line proves nothing about the CLI.
    """
    from agent.lsp import cli as lsp_cli

    svc = make_service(idle_timeout=123.0, max_clients=7)
    inject(svc, "/tmp/ws-render", idle_for=5.0)
    monkeypatch.setattr("agent.lsp.get_service", lambda: svc)
    monkeypatch.setattr("agent.lsp.install.detect_status", lambda pkg: "missing")

    try:
        assert lsp_cli._cmd_status(False) == 0
    finally:
        svc.shutdown()

    out = capsys.readouterr().out
    assert "idle_timeout" in out and "123.0" in out, out
    assert "max_clients" in out and "7" in out, out
    assert "idle=" in out, out


def test_json_status_carries_the_bounds(monkeypatch, capsys):
    """The machine-readable path is the one tooling parses."""
    import json

    from agent.lsp import cli as lsp_cli

    svc = make_service(idle_timeout=123.0, max_clients=7)
    inject(svc, "/tmp/ws-json", idle_for=5.0)
    monkeypatch.setattr("agent.lsp.get_service", lambda: svc)
    monkeypatch.setattr("agent.lsp.install.detect_status", lambda pkg: "missing")

    try:
        assert lsp_cli._cmd_status(True) == 0
    finally:
        svc.shutdown()

    service = json.loads(capsys.readouterr().out)["service"]
    assert service["idle_timeout"] == 123.0
    assert service["max_clients"] == 7
    assert service["clients"][0]["inflight"] == 0
    assert service["clients"][0]["idle_seconds"] >= 5.0


# ----------------------------------------------------------------------
# docs
# ----------------------------------------------------------------------


def test_user_guide_does_not_promise_a_disableable_cap():
    """The docs must not document a knob the code deliberately refuses.

    Ported from #50 as a pure regression guard — the guide is already
    correct.  ``max_clients: 0`` was once documented as the way to
    disable the cap, and the coercion treats 0 as garbage and
    substitutes the host default.  A doc that contradicts the code is
    how an operator "disables" a bound and never learns it is still on.
    """
    from pathlib import Path

    doc = Path(__file__).resolve().parents[3] / (
        "website/docs/user-guide/features/lsp.md"
    )
    text = doc.read_text(encoding="utf-8")

    assert "`max_clients: 0` to disable the cap" not in text
    assert "max_clients" in text, "guard the path, not just the absence"
