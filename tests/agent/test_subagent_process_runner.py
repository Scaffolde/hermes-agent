from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest

import agent.subagent_process_runner as runner_module
from agent.subagent_process_runner import (
    LinuxStrictConfig,
    ProcessRunSpec,
    RuntimeMount,
    StrictPrerequisiteResult,
    _build_linux_strict_argv,
    _bounded_diagnostic_text,
    _classify_terminal_state,
    _minimal_environment,
    cleanup_cgroup,
    probe_linux_strict,
    run_owned_process,
)
from agent.subagent_worker_main import (
    read_authenticated_frame,
    send_authenticated_frame,
)


PYTHON = sys.executable
SECRET = b"c" * 32


def test_process_group_helpers_fail_closed_without_killpg(monkeypatch):
    monkeypatch.delattr(runner_module.os, "killpg")

    assert runner_module._process_group_exists(12345) is False
    assert runner_module._signal_process_group(12345, signal.SIGTERM) is False


def test_bounded_diagnostic_text_strips_control_characters():
    assert (
        _bounded_diagnostic_text("provider\r\n\x1b[31merror\x00", limit=100)
        == "provider [31merror"
    )


def _portable_spec(tmp_path: Path, *argv: str, **overrides: Any) -> ProcessRunSpec:
    values = {
        "executable": PYTHON,
        "argv": tuple(argv),
        "cwd": tmp_path,
        "workspace": tmp_path,
        "capability_secret": SECRET,
        "capability_fd_arg": None,
        "timeout_seconds": 5.0,
        "term_grace_seconds": 0.2,
        "kill_grace_seconds": 1.0,
    }
    values.update(overrides)
    return ProcessRunSpec(**cast(Any, values))


def test_minimal_environment_drops_credentials_proxies_and_python_paths(monkeypatch):
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("SAFE_VALUE", "yes")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("HERMES_PROFILE", "private")
    monkeypatch.setenv("AWS_PROFILE", "production")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setenv("PYTHONPATH", "/ambient/imports")
    monkeypatch.setenv("LD_PRELOAD", "/ambient/injector.so")

    env = _minimal_environment(
        explicit={"WORKER_MODE": "owned"},
        ambient_allowlist=(
            "LANG",
            "SAFE_VALUE",
            "OPENAI_API_KEY",
            "HERMES_PROFILE",
            "AWS_PROFILE",
            "HTTPS_PROXY",
            "NO_PROXY",
            "PYTHONPATH",
            "LD_PRELOAD",
        ),
    )

    assert env == {
        "LANG": "en_US.UTF-8",
        "SAFE_VALUE": "yes",
        "WORKER_MODE": "owned",
    }


def test_explicit_sensitive_environment_is_rejected():
    with pytest.raises(ValueError, match="forbidden environment"):
        _minimal_environment(explicit={"ANTHROPIC_API_KEY": "secret"})


def test_portable_runner_scrubs_real_child_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-pass")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-pass")
    monkeypatch.setenv("PYTHONPATH", "/must/not/pass")
    code = "import json,os; print(json.dumps(dict(os.environ), sort_keys=True))"

    result = run_owned_process(
        _portable_spec(
            tmp_path,
            "-c",
            code,
            environment={"WORKER_MODE": "owned"},
            ambient_env_allowlist=("LANG",),
        )
    )

    child_env = json.loads(result.stdout.decode("utf-8"))
    assert result.state == "SUCCEEDED"
    assert result.confinement == "portable-process-unconfined"
    assert child_env.get("WORKER_MODE") == "owned"
    assert "OPENAI_API_KEY" not in child_env
    assert "HTTPS_PROXY" not in child_env
    assert "PYTHONPATH" not in child_env
    assert result.cleanup.root_reaped is True
    assert result.cleanup.process_group_empty is True


def test_portable_runner_bounds_stdout_and_stderr(tmp_path):
    code = "import os; os.write(1, b'o' * 20000); os.write(2, b'e' * 20000)"
    result = run_owned_process(
        _portable_spec(tmp_path, "-c", code, max_output_bytes=1024)
    )

    assert result.state == "SUCCEEDED"
    assert result.stdout == b"o" * 1024
    assert result.stderr == b"e" * 1024
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


@pytest.mark.live_system_guard_bypass
def test_deadline_escalates_term_then_kill_and_reaps_process_group(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    result = run_owned_process(
        _portable_spec(
            tmp_path,
            "-c",
            code,
            timeout_seconds=0.15,
            term_grace_seconds=0.05,
        )
    )

    assert result.state == "TIMED_OUT"
    assert result.cleanup.term_sent is True
    assert result.cleanup.kill_sent is True
    assert result.cleanup.root_reaped is True
    assert result.cleanup.process_group_empty is True


@pytest.mark.live_system_guard_bypass
def test_deadline_reaps_owned_descendant_process_group(tmp_path):
    child = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(30)"
    )

    result = run_owned_process(
        _portable_spec(
            tmp_path,
            "-c",
            parent,
            timeout_seconds=0.15,
            term_grace_seconds=0.05,
        )
    )

    assert result.state == "TIMED_OUT"
    assert result.cleanup.descendant_scope == "process-group"
    assert result.cleanup.kill_sent is True
    assert result.cleanup.process_group_empty is True


@pytest.mark.live_system_guard_bypass
def test_explicit_cancellation_terminates_and_reaps_process_group(tmp_path):
    cancellation_requested = threading.Event()
    started = threading.Event()
    results = []

    class Receipt:
        def on_created(self, **_kwargs):
            pass

        def on_started(self, **_kwargs):
            started.set()

        def on_terminal(self, *, state, result):
            assert state == "CANCELLED"
            assert result.cleanup.root_reaped is True
            assert result.cleanup.process_group_empty is True

    thread = threading.Thread(
        target=lambda: results.append(
            run_owned_process(
                _portable_spec(
                    tmp_path,
                    "-c",
                    "import time; time.sleep(30)",
                    cancellation_event=cancellation_requested,
                    receipt=Receipt(),
                )
            )
        )
    )
    thread.start()
    assert started.wait(timeout=1)

    cancellation_requested.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(results) == 1
    assert results[0].state == "CANCELLED"
    assert results[0].timed_out is False
    assert results[0].cleanup.term_sent is True


@pytest.mark.parametrize(
    ("cleanup", "diagnostic"),
    [
        (
            runner_module.CleanupEvidence(
                requested=True,
                root_reaped=False,
                process_group_empty=True,
            ),
            "root process was not reaped",
        ),
        (
            runner_module.CleanupEvidence(
                requested=True,
                root_reaped=True,
                process_group_empty=False,
            ),
            "process group was not empty",
        ),
    ],
)
def test_cancellation_fails_closed_when_portable_cleanup_is_incomplete(
    cleanup, diagnostic
):
    state, detail = runner_module._classify_terminal_state(
        returncode=-signal.SIGKILL,
        timed_out=False,
        broker_failed=False,
        cleanup=cleanup,
        cancellation_requested=True,
    )

    assert state == "FAILED"
    assert detail == diagnostic


def test_monitor_is_bounded_even_when_group_signals_fail(monkeypatch):
    class FakeProcess:
        pid = 91919
        returncode = None
        stdout = None
        stderr = None
        killed = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -signal.SIGKILL

    process = FakeProcess()
    monkeypatch.setattr(runner_module, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        runner_module, "_signal_process_group", lambda _pgid, _sig: False
    )
    started = runner_module.time.monotonic()

    (
        _stdout,
        _stderr,
        timed_out,
        _stdout_truncated,
        _stderr_truncated,
        _cancellation_observed,
    ) = runner_module._monitor_process(
        cast(Any, process),
        timeout_seconds=0.02,
        term_grace_seconds=0.02,
        kill_grace_seconds=0.02,
        max_output_bytes=128,
    )

    assert timed_out is True
    assert process.killed is True
    assert runner_module.time.monotonic() - started < 0.5


def test_runner_bootstraps_secret_over_fd_and_broker_uses_authenticated_frames(
    tmp_path,
):
    worker_path = Path(runner_module.__file__).with_name("subagent_worker_main.py")
    observed = []

    class Broker:
        def serve(self, channel: socket.socket, *, root_pid: int, stop_requested):
            observed.append(root_pid)
            send_authenticated_frame(channel, {"operation": "ping"}, SECRET)
            observed.append(read_authenticated_frame(channel, SECRET))

        def cancel(self):
            pass

    result = run_owned_process(
        _portable_spec(
            tmp_path,
            str(worker_path),
            capability_fd_arg="--capability-fd",
            broker=Broker(),
        )
    )

    assert result.state == "SUCCEEDED"
    assert observed[0] == result.root_pid
    assert observed[1] == {"ok": True, "operation": "pong"}


def test_runner_cancels_inflight_broker_operation_before_terminalizing(tmp_path):
    broker_started = threading.Event()
    release = threading.Event()
    cancel_called = threading.Event()
    results = []

    class Broker:
        def serve(self, _channel: socket.socket, *, root_pid: int, stop_requested):
            assert root_pid > 0
            broker_started.set()
            assert release.wait(timeout=2)

        def cancel(self):
            cancel_called.set()
            release.set()

    thread = threading.Thread(
        target=lambda: results.append(
            run_owned_process(_portable_spec(tmp_path, "-c", "pass", broker=Broker()))
        )
    )
    thread.start()
    assert broker_started.wait(timeout=1)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert cancel_called.is_set()
    assert len(results) == 1
    assert results[0].state == "SUCCEEDED"


def test_runner_bounds_join_when_broker_violates_synchronous_cancel_contract(tmp_path):
    broker_started = threading.Event()
    release = threading.Event()

    class Broker:
        def serve(self, _channel: socket.socket, *, root_pid: int, stop_requested):
            assert root_pid > 0
            broker_started.set()
            release.wait(timeout=2)

        def cancel(self):
            pass

    started = runner_module.time.monotonic()
    result = run_owned_process(_portable_spec(tmp_path, "-c", "pass", broker=Broker()))
    elapsed = runner_module.time.monotonic() - started
    release.set()

    assert broker_started.is_set()
    assert elapsed < 1.5
    assert result.state == "FAILED"
    assert "broker thread did not stop" in (result.diagnostic or "")


def test_runner_rejects_broker_without_synchronous_cancel(tmp_path):
    class Broker:
        def serve(self, _channel: socket.socket, *, root_pid: int, stop_requested):
            pass

    with pytest.raises(ValueError, match="synchronous cancellation"):
        run_owned_process(_portable_spec(tmp_path, "-c", "pass", broker=Broker()))


def test_receipt_terminal_callback_waits_for_broker_quiescence(tmp_path):
    broker_started = threading.Event()
    stop_seen = threading.Event()
    release = threading.Event()
    terminal_called = threading.Event()
    results = []

    class Broker:
        def serve(self, _channel: socket.socket, *, root_pid: int, stop_requested):
            assert root_pid > 0
            broker_started.set()
            assert stop_requested.wait(timeout=2)
            stop_seen.set()
            assert release.wait(timeout=2)

        def cancel(self):
            assert release.wait(timeout=2)

    class Receipt:
        def on_created(self, **_kwargs):
            pass

        def on_started(self, **_kwargs):
            pass

        def on_terminal(self, **_kwargs):
            terminal_called.set()

    thread = threading.Thread(
        target=lambda: results.append(
            run_owned_process(
                _portable_spec(
                    tmp_path,
                    "-c",
                    "pass",
                    broker=Broker(),
                    receipt=Receipt(),
                )
            )
        )
    )
    thread.start()
    assert broker_started.wait(timeout=1)
    assert stop_seen.wait(timeout=1)
    assert not terminal_called.is_set()

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert terminal_called.is_set()
    assert len(results) == 1


def test_receipt_callback_observes_created_started_and_terminal(tmp_path):
    events = []

    class Receipt:
        def on_created(self, *, backend, confinement):
            events.append(("created", backend, confinement))

        def on_started(self, *, root_pid):
            events.append(("started", root_pid))

        def on_terminal(self, *, state, result):
            events.append(("terminal", state, result.root_pid))

    result = run_owned_process(
        _portable_spec(tmp_path, "-c", "pass", receipt=Receipt())
    )

    assert events == [
        ("created", "portable", "portable-process-unconfined"),
        ("started", result.root_pid),
        ("terminal", "SUCCEEDED", result.root_pid),
    ]


def test_portable_spawn_is_argv_only_new_session_closed_fds(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        pid = 43210
        returncode = 0
        stdout = None
        stderr = None

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return FakeProcess()

    monkeypatch.setattr(runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner_module, "send_capability_secret", lambda *a: None)
    monkeypatch.setattr(runner_module, "send_authenticated_frame", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module,
        "_monitor_process",
        lambda *a, **k: (b"", b"", False, False, False, False),
    )
    monkeypatch.setattr(
        runner_module,
        "_cleanup_process_group",
        lambda *a, **k: runner_module.CleanupEvidence(
            root_reaped=True, process_group_empty=True
        ),
    )

    spec = _portable_spec(tmp_path, "-c", "pass")
    result = run_owned_process(spec)

    assert result.state == "SUCCEEDED"
    assert captured["argv"] == [PYTHON, "-c", "pass"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["close_fds"] is True
    assert captured["kwargs"]["stdin"] is runner_module.subprocess.DEVNULL
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    rendered_secret = SECRET.decode("ascii")
    assert rendered_secret not in repr(spec)
    assert all(rendered_secret not in value for value in captured["argv"])
    assert all(
        rendered_secret not in value for value in captured["kwargs"]["env"].values()
    )


def test_linux_bwrap_inherits_passed_capability_fd_without_fake_option(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = ProcessRunSpec(
        executable=sys.executable,
        argv=("-c", "pass"),
        cwd=workspace,
        workspace=workspace,
        capability_secret=SECRET,
        backend="linux-strict",
        strict=LinuxStrictConfig(cgroup_parent=tmp_path),
    )
    argv = runner_module._build_linux_strict_argv(
        spec, "/usr/bin/bwrap", capability_fd=9
    )
    assert "--preserve-fds" not in argv
    assert argv[-2:] == ["--capability-fd", "9"]


def test_pre_spawn_receipt_callback_failure_refuses_spawn(monkeypatch, tmp_path):
    class Receipt:
        def on_created(self, **_kwargs):
            raise RuntimeError("recorder unavailable")

    spawned = False

    def fake_popen(*_args, **_kwargs):
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(runner_module.subprocess, "Popen", fake_popen)
    result = run_owned_process(
        _portable_spec(tmp_path, "-c", "pass", receipt=Receipt())
    )
    assert result.state == "FAILED"
    assert result.root_pid is None
    assert spawned is False


def test_post_spawn_receipt_callback_failure_reaps_owned_process(tmp_path, monkeypatch):
    class Receipt:
        def on_created(self, **_kwargs):
            pass

        def on_started(self, **_kwargs):
            raise RuntimeError("recorder unavailable")

    reaped = []

    class FakeProcess:
        pid = 43211
        returncode = None
        stdout = None
        stderr = None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        runner_module.subprocess, "Popen", lambda *a, **k: FakeProcess()
    )
    monkeypatch.setattr(
        runner_module,
        "_cleanup_process_group",
        lambda process, **kwargs: (
            reaped.append(process.pid)
            or runner_module.CleanupEvidence(root_reaped=True, process_group_empty=True)
        ),
    )
    result = run_owned_process(
        _portable_spec(tmp_path, "-c", "import time; time.sleep(30)", receipt=Receipt())
    )
    assert result.state == "FAILED"
    assert result.root_pid is not None
    assert result.cleanup.root_reaped is True
    assert result.cleanup.process_group_empty is True
    assert reaped == [43211]


def test_linux_strict_probe_fails_closed_before_worker_spawn(monkeypatch, tmp_path):
    strict = LinuxStrictConfig(cgroup_parent=tmp_path / "delegated")
    spec = _portable_spec(tmp_path, "-c", "pass", backend="linux-strict", strict=strict)
    monkeypatch.setattr(
        runner_module,
        "probe_linux_strict",
        lambda _config: StrictPrerequisiteResult(
            available=False,
            diagnostics=("bwrap is unavailable",),
        ),
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("worker must not spawn"),
    )

    result = run_owned_process(spec)

    assert result.state == "CONTAINMENT_FAILED"
    assert result.root_pid is None
    assert result.diagnostic == "bwrap is unavailable"


def test_strict_cleanup_failure_supersedes_timeout_terminal_state():
    state, diagnostic = _classify_terminal_state(
        returncode=-signal.SIGKILL,
        timed_out=True,
        broker_failed=False,
        cleanup=runner_module.CleanupEvidence(
            root_reaped=True,
            process_group_empty=True,
            cgroup_empty=False,
        ),
    )

    assert state == "CONTAINMENT_FAILED"
    assert diagnostic == "dedicated cgroup was not empty after cleanup"


def test_broker_failure_supersedes_cancellation_terminal_state():
    state, diagnostic = _classify_terminal_state(
        returncode=-signal.SIGTERM,
        timed_out=False,
        broker_failed=True,
        cleanup=runner_module.CleanupEvidence(
            root_reaped=True,
            process_group_empty=True,
        ),
        cancellation_requested=True,
    )

    assert state == "FAILED"
    assert diagnostic == "broker callback failed"


def test_late_cancellation_does_not_relabel_completed_process(tmp_path, monkeypatch):
    cancellation_requested = threading.Event()
    original_cleanup = runner_module._cleanup_process_group

    def cleanup_then_cancel(*args, **kwargs):
        evidence = original_cleanup(*args, **kwargs)
        cancellation_requested.set()
        return evidence

    monkeypatch.setattr(runner_module, "_cleanup_process_group", cleanup_then_cancel)
    result = run_owned_process(
        _portable_spec(
            tmp_path,
            "-c",
            "pass",
            cancellation_event=cancellation_requested,
        )
    )

    assert result.state == "SUCCEEDED"


def test_strict_argv_validation_precedes_cgroup_and_worker_spawn(monkeypatch, tmp_path):
    missing_runtime = tmp_path / "missing-python"
    spec = ProcessRunSpec(
        executable="/runtime/python",
        argv=(),
        cwd=tmp_path,
        workspace=tmp_path,
        capability_secret=SECRET,
        backend="linux-strict",
        runtime_mounts=(RuntimeMount(missing_runtime, Path("/runtime/python")),),
        strict=LinuxStrictConfig(cgroup_parent=tmp_path / "cgroup"),
    )
    monkeypatch.setattr(
        runner_module,
        "probe_linux_strict",
        lambda _config: StrictPrerequisiteResult(
            available=True,
            bwrap_path="/usr/bin/bwrap",
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_create_owned_cgroup",
        lambda *_a, **_k: pytest.fail("cgroup must not be created"),
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("worker must not spawn"),
    )

    with pytest.raises(ValueError, match="runtime mount source"):
        run_owned_process(spec)


def test_probe_requires_linux_bwrap_cgroup_v2_and_namespaces(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Darwin")
    result = probe_linux_strict(LinuxStrictConfig(cgroup_parent=tmp_path))
    assert result.available is False
    assert "Linux" in result.diagnostics[0]


def test_linux_strict_probe_checks_bwrap_cgroup_and_namespace_support(
    monkeypatch, tmp_path
):
    checked = []
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(runner_module.os, "access", lambda _path, _mode: True)
    monkeypatch.setattr(
        runner_module,
        "_probe_cgroup_delegation",
        lambda parent: checked.append(("cgroup", parent)),
    )
    monkeypatch.setattr(
        runner_module,
        "_probe_namespace_support",
        lambda bwrap, timeout: checked.append(("namespace", bwrap, timeout)),
    )
    config = LinuxStrictConfig(cgroup_parent=tmp_path)

    result = probe_linux_strict(config)

    assert result == StrictPrerequisiteResult(
        available=True,
        bwrap_path="/usr/bin/bwrap",
    )
    assert checked == [
        ("cgroup", tmp_path),
        ("namespace", "/usr/bin/bwrap", 3.0),
    ]


def test_linux_strict_probe_fails_when_namespace_probe_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner_module.os, "access", lambda _path, _mode: True)
    monkeypatch.setattr(runner_module, "_probe_cgroup_delegation", lambda _parent: None)
    monkeypatch.setattr(
        runner_module,
        "_probe_namespace_support",
        lambda _bwrap, _timeout: "user namespace unavailable",
    )

    result = probe_linux_strict(
        LinuxStrictConfig(
            cgroup_parent=tmp_path,
            bwrap_path="/usr/bin/bwrap",
        )
    )

    assert result.available is False
    assert result.diagnostics == ("user namespace unavailable",)


def test_strict_argv_mounts_only_declarations_and_unshares_network(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    python_file = runtime / "python"
    python_file.write_text("runtime", encoding="utf-8")
    spec = ProcessRunSpec(
        executable="/runtime/python",
        argv=("/runtime/worker.py",),
        cwd=workspace,
        workspace=workspace,
        capability_secret=SECRET,
        backend="linux-strict",
        runtime_mounts=(RuntimeMount(python_file, Path("/runtime/python")),),
        strict=LinuxStrictConfig(cgroup_parent=tmp_path / "cgroup"),
    )

    argv = _build_linux_strict_argv(spec, "/usr/bin/bwrap", capability_fd=9)

    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert ["--ro-bind", str(python_file), "/runtime/python"] == argv[
        argv.index("--ro-bind") : argv.index("--ro-bind") + 3
    ]
    bind_index = argv.index("--bind")
    assert argv[bind_index : bind_index + 3] == [
        "--bind",
        str(workspace),
        "/workspace",
    ]
    assert "--tmpfs" in argv and "/tmp" in argv
    assert argv.index("--tmpfs") < bind_index
    home_setenv_index = argv.index("HOME") - 1
    assert argv[home_setenv_index : home_setenv_index + 3] == [
        "--setenv",
        "HOME",
        "/tmp/hermes-worker-home",
    ]
    assert any(
        argv[index : index + 2] == ["--dir", "/tmp/hermes-worker-home"]
        for index in range(len(argv) - 1)
    )
    workspace_dir_index = next(
        index
        for index in range(len(argv) - 1)
        if argv[index : index + 2] == ["--dir", "/workspace"]
    )
    assert argv[workspace_dir_index : workspace_dir_index + 2] == [
        "--dir",
        "/workspace",
    ]
    assert workspace_dir_index < bind_index
    assert "--proc" in argv and "/proc" in argv
    assert "--dev" in argv and "/dev" in argv
    assert argv[-4:] == [
        "/runtime/python",
        "/runtime/worker.py",
        "--capability-fd",
        "9",
    ]


def test_cgroup_cleanup_kills_owned_members_and_proves_empty(monkeypatch, tmp_path):
    cgroup = tmp_path / "owned"
    cgroup.mkdir()
    procs = cgroup / "cgroup.procs"
    procs.write_text("111\n222\n", encoding="ascii")
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))
        current = [
            line
            for line in procs.read_text(encoding="ascii").splitlines()
            if line != str(pid)
        ]
        procs.write_text("\n".join(current), encoding="ascii")

    monkeypatch.setattr(os, "kill", fake_kill)
    evidence = cleanup_cgroup(cgroup, timeout_seconds=0.2)

    assert killed == [(111, signal.SIGKILL), (222, signal.SIGKILL)]
    assert evidence.cgroup_empty is True


def test_cgroup_cleanup_never_claims_empty_when_procs_is_unreadable(tmp_path):
    cgroup = tmp_path / "owned"
    cgroup.mkdir()

    evidence = cleanup_cgroup(cgroup, timeout_seconds=0.1)

    assert evidence.cgroup_empty is False
    assert evidence.cgroup_removed is False
