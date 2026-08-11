"""The test suite must never write into the operator's real Hermes logs.

`hermes_cli/main.py` calls `setup_logging()` at module scope, which resolves
`get_hermes_home()` and attaches rotating file handlers to the ROOT logger.
Importing it - which many test modules do, directly or transitively - wires
the whole pytest session's logging to `<HERMES_HOME>/logs/agent.log`.

If HERMES_HOME is not already sandboxed at that moment, that is the
operator's real log. Measured on a live install, 126 warnings in a personal
`agent.log` came from test runs rather than the running gateway: phantom
`FakeTree` Discord failures and `rejected invalid API key` entries from
`test_api_server_runs.py`. Noise like that makes genuine warnings hard to
find precisely when someone is debugging.

The per-test env fixture cannot close this: fixtures run after collection has
imported the test modules, and by then the handler holds an absolute path.
`tests/conftest.py` sets HERMES_HOME at module scope for that reason - this
guards the property so a refactor cannot quietly undo it.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Set on the child pytest spawned by
#: ``test_preset_home_guard_is_not_inert_under_ci``. The spawning test skips
#: itself when it sees this, so the child can never re-spawn a grandchild no
#: matter which node ids it is pointed at.
_CHILD_SENTINEL = "HERMES_LOG_ISOLATION_NESTED"


def _real_hermes_home() -> Path:
    """Where the operator's logs live, ignoring any test sandboxing."""
    return Path.home() / ".hermes"


def _all_file_destinations() -> list[str]:
    """Every file path the root logger can reach, including via a QueueHandler.

    Logging is routed through a queue, so the file handlers hang off the
    listener rather than the root logger - checking `root.handlers` alone
    reports nothing and looks falsely clean.
    """
    seen: list[str] = []

    def collect(handlers) -> None:
        for handler in handlers or ():
            path = getattr(handler, "baseFilename", None)
            if path:
                seen.append(str(path))
            listener = getattr(handler, "listener", None)
            if listener is not None:
                collect(getattr(listener, "handlers", ()))

    collect(logging.getLogger().handlers)

    try:
        import hermes_logging

        listener = getattr(hermes_logging, "_queue_listener", None)
        if listener is not None:
            collect(getattr(listener, "handlers", ()))
    except Exception:
        pass

    return seen


class TestLogIsolation:
    def test_hermes_home_is_sandboxed_before_imports(self):
        # Deliberately NOT os.environ: by test time the per-test `_isolate_env`
        # fixture has sandboxed HERMES_HOME, so reading it here would pass even
        # with the conftest block deleted. Assert the value captured at conftest
        # import, which is the moment that actually matters.
        from tests.conftest import HERMES_HOME_AT_CONFTEST_IMPORT as home

        assert home, "conftest must set HERMES_HOME before test modules import"
        assert Path(home).resolve() != _real_hermes_home().resolve(), (
            f"HERMES_HOME pointed at the operator's real home ({home}) when "
            "conftest loaded; import-time setup_logging() writes to their agent.log"
        )

    def test_sandbox_overrides_a_preset_hermes_home(self):
        """A preset HERMES_HOME must be replaced, not honored.

        The sandbox used to apply only when HERMES_HOME was unset, so anyone
        running the suite from inside a Hermes agent or gateway process (where
        it is exported and points at the real root) got no sandbox at all and
        wrote straight into the operator's agent.log.
        """
        from tests.conftest import (
            _PRE_SANDBOX_HERMES_HOME,
            _SESSION_HERMES_HOME,
            HERMES_HOME_AT_CONFTEST_IMPORT,
        )

        assert HERMES_HOME_AT_CONFTEST_IMPORT == _SESSION_HERMES_HOME, (
            "conftest must install its own sandbox unconditionally, not defer "
            f"to an inherited HERMES_HOME ({_PRE_SANDBOX_HERMES_HOME!r})"
        )

    def test_kanban_deny_list_still_sees_the_real_root(self):
        """The unconditional sandbox must not blind the kanban write guard.

        The guard is a deny-list anchored on the operator's REAL root. If it
        resolved from the (now always rewired) environment it would point at
        the throwaway tempdir and silently stop protecting ~/.hermes — the
        #69385 regression, reachable again via the sandbox change.
        """
        from tests.conftest import _REAL_KANBAN_ROOT, _SESSION_HERMES_HOME

        sandbox = Path(_SESSION_HERMES_HOME).resolve()
        assert _REAL_KANBAN_ROOT != sandbox
        assert sandbox not in _REAL_KANBAN_ROOT.parents

    def test_preset_home_guard_is_not_inert_under_ci(self):
        """`test_sandbox_overrides_a_preset_hermes_home` must have teeth in CI.

        That guard can only tell the conditional sandbox apart from the
        unconditional one when HERMES_HOME is already set at conftest import.
        Nothing in CI sets it — so with the conditional sandbox restored, the
        `if not os.environ.get("HERMES_HOME")` branch is taken, `_SESSION_HERMES_HOME`
        is bound anyway, and the guard passes on the very code it exists to
        reject. Measured on this tree: reverting conftest to the conditional
        form fails 2 tests with HERMES_HOME preset and 0 with it unset.

        A guard that only fires in an environment CI never reproduces cannot
        stop the regression from being reintroduced. Re-run it in a child
        pytest that does supply the distinguishing condition.
        """
        if os.environ.get(_CHILD_SENTINEL):
            pytest.skip("child of the preset-HERMES_HOME re-run; would recurse")

        decoy = Path(tempfile.mkdtemp(prefix="hermes-preset-home-"))
        (decoy / "logs").mkdir(parents=True, exist_ok=True)
        assert decoy.resolve() != _real_hermes_home().resolve()

        env = {
            **os.environ,
            "HERMES_HOME": str(decoy),
            _CHILD_SENTINEL: "1",
            # The child must not inherit this process's sandbox tempdir, which
            # would mask the preset value we are deliberately supplying.
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        node = (
            "tests/test_log_isolation.py::TestLogIsolation"
            "::test_sandbox_overrides_a_preset_hermes_home"
        )
        child = subprocess.run(
            [sys.executable, "-m", "pytest", node, "-q", "-p", "no:randomly"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            # Bounded: one trivial test. An unbounded spawn here would hang a
            # CI slice for the full job timeout instead of failing the file.
            timeout=300,
        )

        assert child.returncode == 0, (
            "the sandbox guard fails once HERMES_HOME is actually preset, which "
            "is the condition it is written for and the one CI never supplies:\n"
            f"{child.stdout}\n{child.stderr}"
        )

    def test_importing_the_cli_does_not_target_the_real_logs(self):
        pytest.importorskip("hermes_cli.main")

        real_logs = str(_real_hermes_home() / "logs")
        offenders = [p for p in _all_file_destinations() if p.startswith(real_logs)]

        assert offenders == [], (
            "the test session is writing into the operator's real Hermes logs:\n  "
            + "\n  ".join(offenders)
        )
