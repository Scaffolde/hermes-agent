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

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Read at IMPORT time — i.e. during collection, before any fixture has run.
# This is the observation window the SCA-4714 bug lived in, so the value has to
# be sampled here; sampling inside a test body would read the hermetic
# fixture's monkeypatched value and pass no matter what.
_LAZY_INSTALL_FLAG_AT_COLLECTION = os.environ.get("HERMES_DISABLE_LAZY_INSTALLS")


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


@contextlib.contextmanager
def _running_env_pointed_at(venv: Path):
    """Treat *venv* as the environment pytest is executing in.

    The allow-cases below are only controls if the guard's target and the
    running env are the SAME path under the buggy behaviour — otherwise they
    pass whether or not the fix is present, which is exactly what the first
    revision of these tests did.
    """
    conftest = _conftest_module()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "prefix", str(venv))
        mp.setattr(sys, "base_prefix", str(venv))
        mp.setattr(sys, "executable", str(venv / "bin" / "python"))
        conftest._running_env_roots.cache_clear()
        conftest._running_env_root_names.cache_clear()
        yield
    conftest._running_env_roots.cache_clear()
    conftest._running_env_root_names.cache_clear()


def _fake_checkout(tmp_path: Path) -> Path:
    """A project root with a ``pyproject.toml`` and a ``.venv``."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (checkout / ".venv" / "bin").mkdir(parents=True)
    return checkout


def _uv_stub(directory: Path) -> Path:
    """A no-op executable named ``uv``, runnable on this platform.

    The allow-cases have to reach the REAL ``subprocess`` call — that is the
    whole assertion — so the stub must actually execute. An extensionless
    ``#!/bin/sh`` script cannot: Windows ``CreateProcess`` has no shebang
    handling and errors out, which would turn the allow-cases into failures
    that say nothing about the guard. Windows gets a ``.cmd`` shim instead,
    which is also how uv itself is installed there, and which the guard now
    recognises as ``uv``.
    """
    if os.name == "nt":
        stub = directory / "uv.cmd"
        stub.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        stub = directory / "uv"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    return stub


def _run_allowed(argv: list[str], **kwargs) -> None:
    """Assert the guard permits *argv* and the call reaches the real spawn.

    A ``RuntimeError`` here is the guard blocking a safe command — a false
    positive, and a test failure. ``OSError`` is the operating system
    declining to execute the stub, which happens after the guard has already
    allowed the call; on Windows that is not a guard defect, so the assertion
    stops at "not blocked" rather than inventing a weaker probe.
    """
    try:
        completed = subprocess.run(argv, **kwargs)
    except OSError:
        if os.name == "nt":
            return
        raise
    assert completed.returncode == 0


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

    def test_blocks_uv_sync_launched_from_a_subdirectory_of_the_checkout(
        self, fake_running_env
    ):
        """uv discovers the project by walking PARENTS — so must the guard.

        ``uv help sync``: uv searches the current directory and every parent
        for a project. A sync launched from ``<checkout>/docs`` therefore
        resolves to the repository root and prunes the live root ``.venv``.
        Assuming ``<cwd>/.venv`` reads that as ``docs/.venv``, which does not
        exist, and lets the exact corruption of SCA-4712 through.
        """
        checkout = fake_running_env.parent
        (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        venv = checkout / ".venv"
        venv.mkdir()
        subdir = checkout / "docs"
        subdir.mkdir()

        conftest = _conftest_module()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "prefix", str(venv))
            mp.setattr(sys, "base_prefix", str(venv))
            mp.setattr(sys, "executable", str(venv / "bin" / "python"))
            conftest._running_env_roots.cache_clear()
            conftest._running_env_root_names.cache_clear()
            with pytest.raises(RuntimeError, match="RUNNING IN"):
                subprocess.run(["uv", "sync", "--extra", "all"], cwd=str(subdir))
        conftest._running_env_roots.cache_clear()
        conftest._running_env_root_names.cache_clear()

    def test_blocks_uv_sync_pointed_at_the_checkout_by_project_flag(
        self, fake_running_env, tmp_path
    ):
        """``--project`` names the project outright, from anywhere."""
        checkout = fake_running_env.parent
        (checkout / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        venv = checkout / ".venv"
        venv.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        conftest = _conftest_module()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "prefix", str(venv))
            mp.setattr(sys, "base_prefix", str(venv))
            mp.setattr(sys, "executable", str(venv / "bin" / "python"))
            conftest._running_env_roots.cache_clear()
            conftest._running_env_root_names.cache_clear()
            with pytest.raises(RuntimeError, match="RUNNING IN"):
                subprocess.run(
                    ["uv", "sync", "--project", str(checkout)], cwd=str(elsewhere)
                )
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

    def test_allows_a_script_that_only_mentions_a_package_manager_in_text(
        self, fake_running_env, tmp_path
    ):
        """A mention is not an invocation — command position is what counts.

        Regression control for the CI breakage this guard caused on its first
        run: ``tests/test_install_macos_launcher.py`` executes the real
        ``setup_path`` from ``scripts/install.sh``, which *logs* the literal
        text ``uv pip install -e '.[all]'`` and has an English comment about
        "the venv pip entry point". The first revision tokenised the whole
        script and scanned every token, so both were treated as invocations
        and the unrelated test was blocked (CI slices 1/8 and 3/8).

        The script below reproduces all three shapes that broke it: a
        package-manager name inside a double-quoted argument, one inside a
        ``#`` comment, and an apostrophe (``didn't``) that desynchronised the
        old ``shlex`` pass.
        """
        script = (
            'echo "Try: cd /x && uv pip install -e \'.[all]\'"\n'
            "# the venv pip entry point is rewritten here; it didn't exist before\n"
            "echo done\n"
        )
        completed = subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, "VIRTUAL_ENV": str(fake_running_env)},
            capture_output=True,
        )
        assert completed.returncode == 0

    def test_allows_a_sandboxed_target(self, fake_running_env, tmp_path):
        """The behaviour under test is legitimate — only the blast radius isn't.

        A sync aimed at a throwaway venv must still reach the real
        ``subprocess`` call, so every test that drives an install/sync code
        path against ``tmp_path`` keeps working unchanged.
        """
        stub_uv = _uv_stub(tmp_path)
        throwaway = tmp_path / "throwaway-venv"
        (throwaway / "bin").mkdir(parents=True)

        _run_allowed(
            [str(stub_uv), "sync", "--python", str(throwaway / "bin" / "python")],
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(throwaway)},
        )

    def test_allows_read_only_package_commands(self, fake_running_env, tmp_path):
        """``uv pip list`` mutates nothing — blocking it would be a false positive."""
        stub_uv = _uv_stub(tmp_path)

        _run_allowed(
            [str(stub_uv), "pip", "list"],
            env={**os.environ, "VIRTUAL_ENV": str(fake_running_env)},
        )

    def test_allows_uv_syncs_read_only_modes(self, tmp_path):
        """``--check`` and ``--dry-run`` report; they mutate nothing.

        ``uv help sync``: ``--check`` only checks whether the environment is
        synchronized, and ``--dry-run`` explicitly does not modify the
        lockfile or the project environment. Both aim squarely at the RUNNING
        checkout by design — which is what makes this a control: the sync
        resolves to exactly the running env, so a guard that ignored the
        read-only modes would block it.
        """
        checkout = _fake_checkout(tmp_path)
        stub_uv = _uv_stub(tmp_path)

        with _running_env_pointed_at(checkout / ".venv"):
            for read_only in ("--check", "--dry-run"):
                _run_allowed([str(stub_uv), "sync", read_only], cwd=str(checkout))

    def test_allows_a_relative_target_under_the_subprocess_cwd(self, tmp_path):
        """A relative ``--python`` belongs to the CHILD's cwd, not pytest's.

        The control: pytest's own cwd is the checkout whose ``.venv`` IS the
        running env, while the child runs in ``sandbox``. Resolving
        ``.venv/bin/python`` against pytest's cwd therefore lands exactly on
        the running env and blocks; resolving it against the child's cwd
        lands on ``sandbox/.venv`` and is correctly allowed.
        """
        checkout = _fake_checkout(tmp_path)
        stub_uv = _uv_stub(tmp_path)
        sandbox = tmp_path / "sandbox"
        (sandbox / ".venv" / "bin").mkdir(parents=True)

        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(checkout)
            with _running_env_pointed_at(checkout / ".venv"):
                _run_allowed(
                    [
                        str(stub_uv),
                        "pip",
                        "install",
                        "--python",
                        os.path.join(".venv", "bin", "python"),
                        "modal",
                    ],
                    cwd=str(sandbox),
                )


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
    """Pin the specific test that was doing the damage — behaviourally.

    An earlier revision grepped the other test's SOURCE for the literal
    ``sys.modules, "modal"``. That proved nothing: the string matches inside a
    comment, survives being moved after the setup call, and breaks on harmless
    requoting or renaming — pass and fail were both uninformative.

    This drives the real test in a subprocess with the install path sabotaged,
    so the assertion is the one that matters: reaching ``_pip_install`` at all
    is the defect, and the test must pass without reaching it.
    """

    def test_modal_setup_test_never_reaches_the_install_path(self, tmp_path):
        target = (
            "tests/hermes_cli/test_setup.py::"
            "test_modal_setup_persists_direct_mode_when_user_chooses_their_own_account"
        )
        repo_root = Path(__file__).resolve().parents[2]

        # A plugin that makes ANY real install attempt fail loudly instead of
        # starting the managed-runtime repair chain that ends in a venv
        # cut-over. If the SDK stub is reached first — the property under test
        # — nothing calls this and the test passes.
        plugin = tmp_path / "sabotage_install.py"
        plugin.write_text(
            "\n".join(
                [
                    "import pytest",
                    "",
                    "",
                    "@pytest.hookimpl(tryfirst=True)",
                    "def pytest_runtest_setup(item):",
                    "    import hermes_cli.tools_config as tc",
                    "",
                    "    def _forbidden(*args, **kwargs):",
                    "        raise AssertionError(",
                    "            'SCA-4712: reached the REAL _pip_install; "
                    "the modal SDK was not stubbed'",
                    "        )",
                    "",
                    "    tc._pip_install = _forbidden",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                target,
                "-p",
                "no:randomly",
                "-p",
                "sabotage_install",
                "-q",
            ],
            cwd=str(repo_root),
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            "the direct-modal setup test must stub the modal SDK before "
            "setup_terminal_backend runs; without it __import__('modal') "
            "fails and the test drives a REAL _pip_install → managed_uv "
            "runtime repair → live venv cut-over (SCA-4712).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


class TestSessionScopedEnvGuard:
    """SCA-4714 — the teeth above must also be armed OUTSIDE test execution.

    The SCA-4712 teeth live in an autouse *fixture*, so they wrap test
    execution only. A whole-``tests/`` run still mutated the live venv,
    because the installs happen during COLLECTION. Measured from a venv reset
    to ``uv sync --frozen --no-install-project --extra dev`` (167 entries):

        collect tests/agent/test_bedrock_empty_text_blocks.py:20 <module>
          → agent/bedrock_adapter.py:47 (module-level ``ensure()``)
            → uv pip install boto3==1.42.89                     167 → 175

        collect tests/gateway/test_teams.py:173 <module>
          → check_teams_requirements() → ensure_and_bind()
            → uv pip install microsoft-teams-apps aiohttp       175 → 198

    Zero tests executed in that run — it aborts after collection with 25
    errors — so no fixture was ever entered.
    """

    def test_guard_is_armed_before_collection(self):
        """``pytest_configure`` must have wrapped the primitives already.

        This is the whole fix in one assertion: by the time any test body
        runs, the wrappers were installed a collection-phase ago.
        """
        conftest = _conftest_module()
        patched = {
            (module.__name__, attr)
            for module, attr, _ in conftest._SESSION_ENV_GUARD_RESTORE
        }
        for expected in (
            ("subprocess", "run"),
            ("subprocess", "Popen"),
            ("subprocess", "call"),
            ("subprocess", "check_call"),
            ("subprocess", "check_output"),
            ("os", "system"),
            ("os", "popen"),
            ("os", "rename"),
            ("os", "replace"),
            ("shutil", "move"),
            ("shutil", "rmtree"),
        ):
            assert expected in patched, (
                f"{expected} is not session-guarded — a module-scope "
                "lazy install reached through it would mutate the running "
                "venv during collection (SCA-4714)"
            )

    def test_lazy_install_kill_switch_is_set_before_collection(self):
        """The suite's own lazy-install kill switch must cover collection.

        ``HERMES_DISABLE_LAZY_INSTALLS=1`` was set by the hermetic *fixture*,
        so it was off for the entire collection phase — while
        .github/workflows/tests.yml documents the contract as already true
        ("The hermetic test env forbids mid-run pip installs
        (HERMES_DISABLE_LAZY_INSTALLS=1 in tests/conftest.py)"). Every
        collection-time ``ensure()`` therefore ran a real install.

        This closes the class at the production kill switch: ``ensure()``
        raises FeatureUnavailable instead of shelling out at all, so the
        subprocess teeth never have to be the only thing standing between a
        module-scope import and the developer's venv.
        """
        assert _LAZY_INSTALL_FLAG_AT_COLLECTION == "1", (
            "HERMES_DISABLE_LAZY_INSTALLS was "
            f"{_LAZY_INSTALL_FLAG_AT_COLLECTION!r} during collection, not '1' "
            "— a module-scope ensure() can reach a real pip/uv install again "
            "(SCA-4714)"
        )

    def test_guarded_popen_is_still_a_Popen_subclass(self):
        """``isinstance(p, Popen)`` and ``Popen[bytes]`` must keep working."""
        for module, attr, real in _conftest_module()._SESSION_ENV_GUARD_RESTORE:
            if (module.__name__, attr) == ("subprocess", "Popen"):
                assert issubclass(subprocess.Popen, real)
                subprocess.Popen[bytes]  # must not raise
                return
        raise AssertionError("subprocess.Popen was not session-guarded")

    def test_blocks_running_env_install(self, fake_running_env):
        """The bedrock/teams shape: a bare install against the running env."""
        conftest = _conftest_module()
        with pytest.raises(RuntimeError, match="session env guard"):
            conftest._session_env_guard_check_cmd(
                "subprocess.run", ["uv", "pip", "install", "boto3==1.42.89"], {}
            )

    def test_blocks_running_env_replacement(self, fake_running_env):
        conftest = _conftest_module()
        with pytest.raises(RuntimeError, match="session env guard"):
            conftest._session_env_guard_check_path(
                "os.rename", str(fake_running_env), str(fake_running_env) + ".old"
            )

    def test_allows_sandboxed_install(self, fake_running_env, tmp_path):
        """An install aimed at a throwaway venv still goes through."""
        conftest = _conftest_module()
        other = tmp_path / "other-venv"
        (other / "bin").mkdir(parents=True)
        conftest._session_env_guard_check_cmd(
            "subprocess.run",
            [
                "uv",
                "pip",
                "install",
                "boto3",
                "--python",
                str(other / "bin" / "python"),
            ],
            {},
        )

    def test_bypass_marker_disarms_it(self, fake_running_env):
        """``live_system_guard_bypass`` must still mean what it meant.

        The session guard is armed outside every fixture, so an opt-out the
        test already declared would be silently overruled unless the fixture
        disarms it explicitly.
        """
        conftest = _conftest_module()
        conftest._ENV_GUARD_BYPASS[0] = True
        try:
            conftest._session_env_guard_check_cmd(
                "subprocess.run", ["uv", "pip", "install", "boto3"], {}
            )
        finally:
            conftest._ENV_GUARD_BYPASS[0] = False

    def test_module_scope_install_is_blocked_during_collection(self, tmp_path):
        """End-to-end control for the window the fixture cannot cover.

        A generated test module runs a package-manager command at MODULE
        scope, so it fires while pytest is importing the module — i.e. during
        collection, before any fixture exists. The command is a no-op ``uv``
        stub, so a RED guard here installs nothing; the control asserts on the
        guard's own message, not on side effects.

        The module has to live under ``tests/`` because ``tests/conftest.py``
        is only loaded for files below it — a tmp_path module would not load
        the guard at all and the control would test nothing.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        bindir = tmp_path / "bin"
        bindir.mkdir()
        stub = _uv_stub(bindir)

        probe = repo_root / "tests" / "test_sca4714_collection_probe_tmp.py"
        probe.write_text(
            "\n".join(
                [
                    '"""Generated by test_running_env_mutation_guard.py; deleted after."""',
                    "import subprocess",
                    "",
                    "subprocess.run(",
                    f"    [{str(stub)!r}, 'pip', 'install', 'hermes-sca4714-probe'],",
                    "    check=False,",
                    ")",
                    "",
                    "",
                    "def test_placeholder():",
                    "    pass",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", str(probe), "-p", "no:randomly", "-q"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
        finally:
            probe.unlink(missing_ok=True)

        output = completed.stdout + completed.stderr
        assert "session env guard" in output, (
            "a module-scope package-manager call against the running env was "
            "NOT blocked during collection — the SCA-4714 window is open "
            f"again.\nexit={completed.returncode}\n{output}"
        )
