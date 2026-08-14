"""Config-bound resolution and the operator-visible report of it.

Two knobs bound the language-server fleet: ``max_clients`` bounds how
*many* live at once, ``idle_timeout`` bounds how *long* each survives.
``resolve_max_clients`` has always rejected unusable input; its sibling
``idle_timeout`` was parsed inline and rejected almost none of it.

``.nan`` and ``.inf`` are valid YAML that survive ``float()`` without
raising, so neither reached the ``except (TypeError, ValueError)`` arm,
and both slipped the ``0 < x < MIN_IDLE_TIMEOUT`` clamp because every
comparison against NaN is False.  The result was a silently disabled
reaper: at ``nan`` the reaper task never starts, and at ``inf`` it
starts and reaps nothing, because its cutoff (``now - idle_timeout``)
is ``-inf`` and no client is ever older than that.  Neither is a crash
and neither is unbounded accumulation -- the count cap still binds --
which is exactly why it went unnoticed: the fleet just pins at
``max_clients`` forever with the reaper reporting itself as on.

These pin both bounds to the same resolver discipline, and pin the
status surface that is how an operator sees which bounds are in force
without reading source.
"""
from __future__ import annotations

import logging
import math
import sys
import types

import pytest

from agent.lsp import manager as manager_mod
from agent.lsp.manager import (
    DEFAULT_IDLE_TIMEOUT,
    MIN_IDLE_TIMEOUT,
    LSPService,
    resolve_idle_timeout,
    resolve_max_clients,
)


def _service(**kw) -> LSPService:
    """A service with no background loop and no real subprocesses."""
    kw.setdefault("enabled", False)
    kw.setdefault("wait_mode", "document")
    kw.setdefault("wait_timeout", 5.0)
    kw.setdefault("install_strategy", "auto")
    kw.setdefault("memory_budget", None)
    return LSPService(**kw)


def _from_config(monkeypatch, lsp_cfg: dict) -> LSPService:
    """Drive the real ``create_from_config`` over a supplied config."""
    fake = types.ModuleType("hermes_cli.config")
    fake.load_config_readonly = lambda: {"lsp": {"enabled": False, **lsp_cfg}}
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake)
    svc = LSPService.create_from_config()
    assert svc is not None
    return svc


# ---------------------------------------------------------------------
# 1. non-finite bounds fall back rather than silently disabling a bound
# ---------------------------------------------------------------------

# Both the float and the string form: YAML yields the float, an env var
# or a hand-edited value yields the string, and both reach the parse.
UNUSABLE = [
    float("nan"),
    float("inf"),
    float("-inf"),
    ".nan",
    ".inf",
    "nan",
    "inf",
    "-inf",
    "nonsense",
    -5,
    -0.5,
    None,
]


@pytest.mark.parametrize("raw", UNUSABLE)
def test_non_finite_bounds_fall_back_instead_of_crashing_the_lsp_path(raw):
    """Neither bound may accept a value that silently switches it off."""
    assert resolve_idle_timeout(raw) == DEFAULT_IDLE_TIMEOUT
    # The sibling knob already behaved; this pins them together so the
    # two cannot drift apart again.
    assert resolve_max_clients(raw) == manager_mod.default_max_clients()


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), ".inf", "nonsense", -5])
def test_unusable_idle_timeout_leaves_the_reaper_armed(monkeypatch, raw):
    """The end the guard exists for: the reaper still starts and still reaps.

    Asserted on the two expressions that actually decide it -- the
    start gate and the reap cutoff -- rather than on the parsed value.
    """
    svc = _from_config(monkeypatch, {"idle_timeout": raw})

    assert math.isfinite(svc._idle_timeout)
    assert svc._idle_timeout > 0, "start gate `_idle_timeout > 0` must fire"
    cutoff = 1_000_000.0 - svc._idle_timeout
    assert math.isfinite(cutoff), "a non-finite cutoff reaps nothing forever"


def test_direct_construction_is_guarded_too():
    """The guard lives at the assignment, not only in the config reader.

    ``_max_clients`` is resolved in ``__init__``; ``_idle_timeout`` must
    be too, or every non-config caller keeps the old hole.
    """
    svc = _service(idle_timeout=float("nan"))
    assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT


# ---------------------------------------------------------------------
# 2. the values that must keep working
# ---------------------------------------------------------------------


def test_zero_still_disables_the_reaper_deliberately():
    """``0`` is the documented off switch and is not garbage."""
    assert resolve_idle_timeout(0) == 0.0
    assert resolve_idle_timeout("0") == 0.0


def test_a_too_small_timeout_is_clamped_not_rejected():
    """Below the per-op wait budget a client can be reaped mid-flight."""
    assert resolve_idle_timeout(1) == MIN_IDLE_TIMEOUT
    assert resolve_idle_timeout(MIN_IDLE_TIMEOUT - 0.001) == MIN_IDLE_TIMEOUT


def test_the_floor_is_config_hygiene_and_does_not_bind_in_process():
    """The two responsibilities have different scopes, on purpose.

    Rejecting unusable input is a safety invariant and holds
    everywhere.  The ``MIN_IDLE_TIMEOUT`` floor guards an operator's
    ``config.yaml`` only -- in-process callers pass a sub-floor timeout
    to drive the reaper loop deterministically, and clamping them broke
    exactly that (``test_reaper_survives_sweep_error``).
    """
    assert resolve_idle_timeout(0.1, clamp_floor=False) == 0.1
    assert _service(idle_timeout=0.1)._idle_timeout == 0.1
    # ...but the safety half still binds on that same path.
    assert _service(idle_timeout=float("inf"))._idle_timeout == DEFAULT_IDLE_TIMEOUT


def test_a_usable_timeout_survives_untouched():
    assert resolve_idle_timeout(45) == 45.0
    assert resolve_idle_timeout("120.5") == 120.5


def test_config_max_clients_zero_is_rejected_loudly(monkeypatch, caplog):
    """An overridden cap must leave a visible trace, not a debug line."""
    with caplog.at_level(logging.WARNING, logger=manager_mod.logger.name):
        svc = _from_config(monkeypatch, {"max_clients": 0})
    assert svc._max_clients == manager_mod.default_max_clients()
    assert any(
        "max_clients" in r.message or "max_clients" in str(r.args)
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ), "operator gets no signal their cap was overridden"


def test_config_idle_timeout_rejection_is_reported(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=manager_mod.logger.name):
        svc = _from_config(monkeypatch, {"idle_timeout": ".inf"})
    assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
    assert any(
        "idle_timeout" in r.message or "idle_timeout" in str(r.args)
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


# ---------------------------------------------------------------------
# 3. the bounds are visible to an operator
# ---------------------------------------------------------------------


def test_status_reports_the_bounds():
    """``hermes lsp status --json`` must show which bounds are in force."""
    svc = _service(idle_timeout=45, max_clients=3)
    status = svc.get_status()
    assert status["max_clients"] == 3
    assert status["idle_timeout"] == 45.0


def test_status_reports_per_client_idle_and_inflight():
    svc = _service(idle_timeout=45, max_clients=3)
    key = ("pyright", "/w")

    class _C:
        state = "ready"
        is_running = True

    svc._clients[key] = _C()
    svc._last_used[key] = manager_mod._idle_clock() - 12.0
    svc._inflight[key] = 2

    entry = next(c for c in svc.get_status()["clients"] if c["server_id"] == "pyright")
    assert entry["inflight"] == 2
    assert entry["idle_seconds"] == pytest.approx(12.0, abs=5.0)


def test_status_idle_seconds_is_none_for_a_never_used_client():
    svc = _service()
    key = ("pyright", "/w")

    class _C:
        state = "ready"
        is_running = True

    svc._clients[key] = _C()
    entry = svc.get_status()["clients"][0]
    assert entry["idle_seconds"] is None
    assert entry["inflight"] == 0


def test_human_readable_status_renders_the_bounds(capsys, monkeypatch):
    """The plain-text branch, not just the JSON one."""
    from agent.lsp import cli as lsp_cli

    svc = _service(idle_timeout=45, max_clients=3)
    monkeypatch.setattr(lsp_cli, "get_service", lambda: svc, raising=False)
    monkeypatch.setattr("agent.lsp.get_service", lambda: svc, raising=False)

    assert lsp_cli._cmd_status(emit_json=False) == 0
    out = capsys.readouterr().out
    assert "max_clients" in out
    assert "3" in out
    assert "idle_timeout" in out
    assert "45" in out
