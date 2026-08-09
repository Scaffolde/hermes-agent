import asyncio
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from agent.google_antigravity_cli_adapter import (
    AntigravityCLIError,
    AsyncGoogleAntigravityCLIClient,
    GoogleAntigravityCLIClient,
    build_prompt_from_messages,
)


def test_build_prompt_from_messages_labels_roles():
    prompt = build_prompt_from_messages([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Summarize this."},
        {"role": "assistant", "content": "Draft."},
    ])

    assert "System:\nBe brief." in prompt
    assert "User:\nSummarize this." in prompt
    assert "Assistant:\nDraft." in prompt


def test_create_invokes_agy_print(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "OK\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Result()

    monkeypatch.setattr("agent.google_antigravity_cli_adapter.subprocess.run", fake_run)
    client = GoogleAntigravityCLIClient(command="/bin/agy", args=["--sandbox"], print_timeout="60s")

    response = client.chat.completions.create(
        model="antigravity-cli",
        messages=[{"role": "user", "content": "Reply OK"}],
    )

    assert response.choices[0].message.content == "OK"
    cmd, kwargs = calls[0]
    assert cmd[0] == "/bin/agy"
    assert cmd[1] == "--sandbox"
    assert "-p" in cmd
    assert "User:\nReply OK" in cmd
    assert cmd[-2:] == ["--print-timeout", "60s"]
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 330.0


def test_create_forwards_call_timeout_to_subprocess(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return CompletedProcess(cmd, 0, stdout="OK", stderr="")

    monkeypatch.setattr("agent.google_antigravity_cli_adapter.subprocess.run", fake_run)
    client = GoogleAntigravityCLIClient(command="agy")

    client.chat.completions.create(
        messages=[{"role": "user", "content": "x"}], timeout=0.25
    )

    assert captured["timeout"] == 0.25


def test_subprocess_timeout_becomes_adapter_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr("agent.google_antigravity_cli_adapter.subprocess.run", fake_run)
    client = GoogleAntigravityCLIClient(command="agy")

    with pytest.raises(AntigravityCLIError, match="timed out after 0.25s"):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "x"}], timeout=0.25
        )


def test_nonzero_exit_raises_redacted_error(monkeypatch):
    class Result:
        returncode = 7
        stdout = ""
        stderr = "failure text"

    monkeypatch.setattr(
        "agent.google_antigravity_cli_adapter.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    client = GoogleAntigravityCLIClient(command="agy")

    with pytest.raises(AntigravityCLIError) as exc:
        client.chat.completions.create(messages=[{"role": "user", "content": "x"}])

    assert "failure text" in str(exc.value)
    assert "exit 7" in str(exc.value)


def test_empty_stdout_success_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "agent.google_antigravity_cli_adapter.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout="", stderr=""),
    )

    client = GoogleAntigravityCLIClient(command="/bin/agy")
    with pytest.raises(AntigravityCLIError, match="produced no output"):
        client.complete("Reply OK")


def test_async_wrapper_returns_completion(monkeypatch):
    class Result:
        returncode = 0
        stdout = "ASYNC OK\n"
        stderr = ""

    monkeypatch.setattr(
        "agent.google_antigravity_cli_adapter.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    sync_client = GoogleAntigravityCLIClient(command="agy")
    async_client = AsyncGoogleAntigravityCLIClient(sync_client)

    response = asyncio.run(async_client.chat.completions.create(
        messages=[{"role": "user", "content": "x"}],
    ))

    assert response.choices[0].message.content == "ASYNC OK"
