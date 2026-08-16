"""The direct-multi-file-invocation warning must fire exactly when it should.

Test isolation in this repo lives in the RUNNER (``scripts/run_tests_parallel.py``
spawns one pytest subprocess per file), not in ``tests/conftest.py``. So a plain
``pytest tests/<dir>`` still runs — it just shares one interpreter across every
file, where module-level state leaks. On ``tests/hermes_cli`` that produced 141
failures on a clean ``main``, none reproducible per-file and none of which CI
can ever see (SCA-4692).

The warning is what stops the next person reading that as a real red. These
tests pin its three cases, plus the constant the runner and conftest must agree
on — if they drift, the marker never arrives and every runner subprocess starts
printing the warning it is supposed to suppress.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import PER_FILE_ISOLATION_ENV, _warn_on_direct_multi_file_run

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_runner():
    """Import ``scripts/run_tests_parallel.py`` (not an installed module)."""
    path = REPO_ROOT / "scripts" / "run_tests_parallel.py"
    spec = importlib.util.spec_from_file_location("_sca4692_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Reporter:
    def __init__(self):
        self.lines = []

    def write_sep(self, *args, **kwargs):
        self.lines.append("SEP")

    def write_line(self, line):
        self.lines.append(line)


def _config(reporter, args=("tests/hermes_cli",)):
    return SimpleNamespace(
        args=list(args),
        pluginmanager=SimpleNamespace(getplugin=lambda name: reporter),
    )


def _items(*files):
    return [SimpleNamespace(location=(f, 1, "test_x")) for f in files]


def _warned(reporter):
    return any("share ONE interpreter" in line for line in reporter.lines)


def test_warns_on_multiple_files(monkeypatch):
    monkeypatch.delenv(PER_FILE_ISOLATION_ENV, raising=False)
    reporter = _Reporter()
    _warn_on_direct_multi_file_run(_config(reporter), _items("a.py", "b.py"))
    assert _warned(reporter)


def test_silent_on_single_file(monkeypatch):
    """``pytest tests/foo.py`` matches the runner's own boundary — no warning."""
    monkeypatch.delenv(PER_FILE_ISOLATION_ENV, raising=False)
    reporter = _Reporter()
    _warn_on_direct_multi_file_run(_config(reporter), _items("a.py", "a.py"))
    assert not _warned(reporter)


def test_silent_under_the_canonical_runner(monkeypatch):
    """The runner sets the marker and gives each child ONE file — stay quiet."""
    monkeypatch.setenv(PER_FILE_ISOLATION_ENV, "1")
    reporter = _Reporter()
    _warn_on_direct_multi_file_run(_config(reporter), _items("a.py", "a.py"))
    assert not reporter.lines


def test_marker_does_not_excuse_a_multi_file_interpreter(monkeypatch):
    """The marker is a promise of one file, not a blanket mute.

    Trusting it over the observed file count is what let a path smuggled
    through the runner's passthrough execute two files in one interpreter
    with no warning at all (SCA-4692 review).
    """
    monkeypatch.setenv(PER_FILE_ISOLATION_ENV, "1")
    reporter = _Reporter()
    _warn_on_direct_multi_file_run(_config(reporter), _items("a.py", "b.py"))
    assert any("share THIS" in line for line in reporter.lines)
    # It must be named as a runner bug, not blamed on the operator.
    assert any("runner bug" in line for line in reporter.lines)


def test_names_the_canonical_runner_in_the_message(monkeypatch):
    """A warning that doesn't say what to run instead just adds noise."""
    monkeypatch.delenv(PER_FILE_ISOLATION_ENV, raising=False)
    reporter = _Reporter()
    _warn_on_direct_multi_file_run(_config(reporter), _items("a.py", "b.py"))
    assert any("scripts/run_tests.sh" in line for line in reporter.lines)


def test_survives_a_missing_terminal_reporter(monkeypatch):
    """``-p no:terminal`` / embedding harnesses must not crash collection."""
    monkeypatch.delenv(PER_FILE_ISOLATION_ENV, raising=False)
    config = SimpleNamespace(
        args=["tests/"],
        pluginmanager=SimpleNamespace(getplugin=lambda name: None),
    )
    _warn_on_direct_multi_file_run(config, _items("a.py", "b.py"))


def test_runner_and_conftest_agree_on_the_marker():
    """Drift here silently re-enables the warning inside every runner child."""
    assert _load_runner().PER_FILE_ISOLATION_ENV == PER_FILE_ISOLATION_ENV


def test_runner_exports_the_marker_to_children(tmp_path, monkeypatch):
    """The marker must actually reach the child process env.

    Behavioral, not textual: drive ``_run_one_file_once`` with a stubbed
    ``subprocess.Popen`` and read the ``env`` it was handed. An assertion on
    the runner's source text would pass while the marker never reached the
    child, and would break on any harmless refactor — the repo bans reading
    source in tests for exactly that pair of reasons.
    """
    runner = _load_runner()
    captured = {}

    class _FakeProc:
        # A pid that cannot exist, so the runner's getpgid lookup fails and it
        # records pgid=None. Never 0: _kill_tree would then killpg(0, SIGKILL)
        # and take out this very test process's own group.
        pid = 2**31 - 1
        returncode = 0

        def communicate(self, timeout=None):
            return ("1 passed in 0.01s", None)

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(runner.subprocess, "Popen", _fake_popen)
    # No real child exists, so there is nothing to reap; neutralise the kill
    # path entirely rather than letting it aim at a pid we made up.
    monkeypatch.setattr(runner, "_kill_tree", lambda *a, **k: None)

    runner._run_one_file_once(
        Path("tests/test_conftest_multifile_warning.py"), [], tmp_path, 60.0
    )

    assert captured["env"] is not None, "runner must hand the child an explicit env"
    assert captured["env"].get(PER_FILE_ISOLATION_ENV) == "1"
    # The child must still inherit the ambient environment, not only the marker.
    assert len(captured["env"]) > 1


@pytest.mark.parametrize("count", [2, 5, 57])
def test_reports_the_file_count(monkeypatch, count):
    monkeypatch.delenv(PER_FILE_ISOLATION_ENV, raising=False)
    reporter = _Reporter()
    files = [f"f{i}.py" for i in range(count)]
    _warn_on_direct_multi_file_run(_config(reporter), _items(*files))
    assert any(f"{count} test files" in line for line in reporter.lines)


# ── The runner must not smuggle a second path into a per-file child ─────────
#
# End-to-end through the real CLI, because the defect lived in argv handling:
# ``run_tests_parallel.py A.py -- B.py`` discovered "1 file" and then executed
# 22 tests from both files in one interpreter, marker set, warning silent.
#
# These drive the runner over THROWAWAY test files in tmp_path, never over this
# repo's own tests. Pointing it at this file would make the spawned child
# re-collect these very tests and spawn again — unbounded recursion the moment
# the guard under test is absent, which is exactly when the test must fail
# cleanly instead.


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tests_parallel.py"), *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture
def two_throwaway_test_files(tmp_path):
    first = tmp_path / "test_sca4692_first.py"
    second = tmp_path / "test_sca4692_second.py"
    first.write_text("def test_first():\n    assert True\n")
    second.write_text("def test_second():\n    assert True\n")
    return first, second


def test_cli_rejects_a_test_path_in_the_passthrough(two_throwaway_test_files):
    first, second = two_throwaway_test_files
    proc = _run_cli(str(first), "--", str(second))

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "not allowed in the pytest passthrough" in proc.stderr
    # The rejection must name the offending path and the correct way to pass it.
    assert str(second) in proc.stderr
    assert "positional" in proc.stderr


def test_cli_keeps_ordinary_passthrough_flags_working(two_throwaway_test_files):
    """The guard rejects paths only — flags and their values must survive."""
    first, _ = two_throwaway_test_files
    proc = _run_cli(str(first), "-k", "first", "--", "--tb=long")

    assert "not allowed in the pytest passthrough" not in proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr
