"""The test suite must never mutate the environment it is RUNNING IN.

SCA-4712. Measured on main: a whole-tree ``pytest tests/hermes_cli`` run
replaced the checkout's live ``.venv`` mid-run, so

  * ``pytest`` vanished from site-packages and a second run could not start
    ("No module named pytest" — which reads as a broken checkout, not as
    test-induced damage);
  * the remaining ~47% of that same run executed against a *different*
    environment than the first half;
  * every uncontrolled before/after comparison taken across two runs was
    silently measuring two different environments.

The chain, captured by instrumenting every subprocess spawn and stat'ing a
site-packages sentinel after each test:

    test_setup.py::test_modal_setup_persists_direct_mode_...
      → hermes_cli.setup.setup_terminal_backend (direct-modal branch)
      → __import__("modal") raises ImportError (nothing stubbed it)
      → tools_config._pip_install(["modal"])
      → managed_uv.ensure_uv() → managed-runtime repair
      → uv sync --extra all --locked  (builds a candidate venv)
      → managed_uv._cut_over_candidate: RENAMES the live .venv aside and
        promotes the candidate — which has no ``dev`` extra, hence no pytest.

CI never sees it: each shard gets a fresh environment and never runs a second
whole-tree pass in a mutated venv. Same blind-spot class as SCA-4692.

These are positive controls for the two guard teeth in ``tests/conftest.py``.
Each asserts the guard actually FIRES — a guard that blocked nothing would
pass a "nothing was mutated" assertion trivially.

Blast radius: the block-cases redirect the guard's notion of "the running
environment" at a ``tmp_path`` venv first (``_running_env_roots`` is
lru_cached, so the patch needs a ``cache_clear``). A RED guard therefore
damages a tmpdir, never the real ``.venv`` — a positive control that can
destroy the developer's environment when it fails is not a control anyone
will keep running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _conftest_module():
    """The exact ``tests/conftest.py`` module object pytest loaded.

    A plain ``import conftest`` would build a SECOND module object with its
    own ``lru_cache``, so clearing that cache would leave the guard's real
    cache untouched and the controls would silently test nothing.
    """
    target = os.path.realpath(
        str(Path(__file__).resolve().parent.parent / "conftest.py")
    )
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and os.path.realpath(path) == target:
            return module
    raise AssertionError("tests/conftest.py is not in sys.modules")


@pytest.fixture
def fake_running_env(tmp_path, monkeypatch):
    """Point the guard at a throwaway venv and yield it.

    Exercises the real guard closure — it reads ``_running_env_roots()`` on
    every call — while keeping the real ``.venv`` outside the blast radius.
    """
    conftest = _conftest_module()
    root = tmp_path / "fake-venv"
    bindir = root / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    python = bindir / ("python.exe" if os.name == "nt" else "python")
    python.write_text("", encoding="utf-8")
    (root / "lib").mkdir()

    monkeypatch.setattr(sys, "prefix", str(root))
    monkeypatch.setattr(sys, "base_prefix", str(root))
    monkeypatch.setattr(sys, "executable", str(python))
    conftest._running_env_roots.cache_clear()
    conftest._running_env_root_names.cache_clear()
    try:
        yield root
    finally:
        # monkeypatch restores sys.* after this fixture; the caches must not
        # keep serving the fake roots to the next test.
        conftest._running_env_roots.cache_clear()
        conftest._running_env_root_names.cache_clear()


class TestPackageManagerTooth:
    """``uv``/``pip`` commands whose resolved target is the running env."""

    def test_blocks_uv_pip_install_into_the_running_venv(self, fake_running_env):
        # The exact argv tools_config._pip_install issues: uv pip install with
        # VIRTUAL_ENV pointed at the running interpreter's venv.
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            subprocess.run(
                ["uv", "pip", "install", "modal"],
                env={**os.environ, "VIRTUAL_ENV": str(fake_running_env)},
                capture_output=True,
            )

    def test_blocks_untargeted_uv_sync_at_the_project_root(self, fake_running_env):
        # `uv sync` with no --python / UV_PROJECT_ENVIRONMENT syncs
        # <cwd>/.venv — and prunes it to the requested extras.
        checkout = fake_running_env.parent
        venv = checkout / ".venv"
        venv.mkdir()
        conftest = _conftest_module()
        conftest._running_env_roots.cache_clear()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "prefix", str(venv))
            mp.setattr(sys, "base_prefix", str(venv))
            mp.setattr(sys, "executable", str(venv / "bin" / "python"))
            conftest._running_env_roots.cache_clear()
            conftest._running_env_root_names.cache_clear()
            with pytest.raises(RuntimeError, match="RUNNING IN"):
                subprocess.run(["uv", "sync", "--extra", "all"], cwd=str(checkout))
        conftest._running_env_roots.cache_clear()
        conftest._running_env_root_names.cache_clear()

    def test_blocks_pip_install_targeting_the_running_interpreter(
        self, fake_running_env
    ):
        python = fake_running_env / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            subprocess.run([str(python), "-m", "pip", "install", "modal"])

    def test_blocks_wrapped_invocations(self, fake_running_env):
        """``bash -c "uv sync"`` must not launder the mutation past the guard."""
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            subprocess.run(
                ["bash", "-c", "uv pip install modal"],
                env={**os.environ, "VIRTUAL_ENV": str(fake_running_env)},
            )

    def test_allows_a_sandboxed_target(self, fake_running_env, tmp_path):
        """The behaviour under test is legitimate — only the blast radius isn't.

        A sync aimed at a throwaway venv must still reach the real
        ``subprocess`` call, so every test that drives an install/sync code
        path against ``tmp_path`` keeps working unchanged.
        """
        stub_uv = tmp_path / "uv"
        stub_uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub_uv.chmod(0o755)
        throwaway = tmp_path / "throwaway-venv"
        (throwaway / "bin").mkdir(parents=True)

        completed = subprocess.run(
            [str(stub_uv), "sync", "--python", str(throwaway / "bin" / "python")],
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(throwaway)},
        )
        assert completed.returncode == 0

    def test_allows_read_only_package_commands(self, fake_running_env, tmp_path):
        """``uv pip list`` mutates nothing — blocking it would be a false positive."""
        stub_uv = tmp_path / "uv"
        stub_uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub_uv.chmod(0o755)

        completed = subprocess.run(
            [str(stub_uv), "pip", "list"],
            env={**os.environ, "VIRTUAL_ENV": str(fake_running_env)},
        )
        assert completed.returncode == 0


class TestEnvironmentReplacementTooth:
    """The venv cut-over path — a rename, not a package command.

    This is the tooth that actually catches SCA-4712: the ``uv sync`` in
    ``_stage_candidate_venv`` legitimately targets the candidate, so the
    package-manager tooth never sees it. The damage is the rename afterwards.
    """

    def test_blocks_parking_the_running_venv(self, fake_running_env, tmp_path):
        # _cut_over_candidate's first move.
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            os.rename(fake_running_env, tmp_path / "parked")

    def test_blocks_promoting_a_candidate_over_the_running_venv(
        self, fake_running_env, tmp_path
    ):
        # ...and its second move.
        candidate = tmp_path / "venv-candidate"
        candidate.mkdir()
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            os.rename(candidate, fake_running_env)

    def test_blocks_pathlib_rename(self, fake_running_env, tmp_path):
        # managed_uv uses Path.rename, which routes through os.rename on 3.11+.
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            fake_running_env.rename(tmp_path / "parked")

    def test_blocks_os_replace(self, fake_running_env, tmp_path):
        candidate = tmp_path / "candidate2"
        candidate.mkdir()
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            os.replace(candidate, fake_running_env)

    def test_blocks_shutil_move(self, fake_running_env, tmp_path):
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            shutil.move(str(fake_running_env), str(tmp_path / "moved"))

    def test_blocks_rmtree(self, fake_running_env):
        with pytest.raises(RuntimeError, match="RUNNING IN"):
            shutil.rmtree(fake_running_env)

    def test_allows_renames_elsewhere(self, fake_running_env, tmp_path):
        src = tmp_path / "unrelated"
        src.mkdir()
        os.rename(src, tmp_path / "unrelated-renamed")
        assert (tmp_path / "unrelated-renamed").is_dir()

    def test_allows_rmtree_elsewhere(self, fake_running_env, tmp_path):
        victim = tmp_path / "scratch"
        victim.mkdir()
        shutil.rmtree(victim)
        assert not victim.exists()


class TestCulpritIsIsolated:
    """Pin the specific test that was doing the damage.

    Not an assertion that "nothing happened" — it pins the SDK stub, which is
    what keeps ``setup_terminal_backend`` out of the install branch entirely.
    """

    def test_modal_test_stubs_the_sdk_before_driving_setup(self):
        source = Path(__file__).with_name("test_setup.py").read_text(encoding="utf-8")
        start = source.index(
            "def test_modal_setup_persists_direct_mode_when_user_chooses_their_own_account"
        )
        end = source.index("def test_vercel_setup_configures_access_token_auth", start)
        body = source[start:end]
        assert 'sys.modules, "modal"' in body, (
            "the direct-modal setup test must stub sys.modules['modal']; "
            "without it __import__('modal') fails and the test runs a REAL "
            "_pip_install → managed_uv runtime repair → live venv cut-over"
        )
