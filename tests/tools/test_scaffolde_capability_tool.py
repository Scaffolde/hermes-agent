import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "scaffolde").mkdir(parents=True)
    bin_dir = home / "test-bin"
    bin_dir.mkdir()
    bun = bin_dir / "bun"
    bun.write_text("#!/bin/sh\nexec python3 \"$@\"\n", encoding="utf-8")
    bun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _write_fake_runner(home: Path, *, exit_code: int = 0) -> Path:
    runner = home / "fake_runner.ts"
    runner.write_text(
        "import json, os, sys\n"
        "keys = ['HOME','PATH','LIFEOS_DIR','GOOGLE_KEYCHAIN_ACCOUNT','SCAFFOLDE_AUTOMATION_SENDER_EMAIL','GOOGLE_APPLICATION_CREDENTIALS']\n"
        "env = {key: os.environ.get(key) for key in keys}\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': env}))\n"
        "print('stderr contains SECRET_TOKEN=should-redact and harmless detail', file=sys.stderr)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return runner


def _descriptor(home: Path, *, runner: Path | None = None, risk="read", extra=None) -> dict:
    runner = runner or _write_fake_runner(home)
    runner_arg = str(runner)
    desc = {
        "version": 1,
        "id": "scaffolde.gmail.pai",
        "tool_name": "scaffolde_gmail",
        "authority": "scaffolde",
        "kind": "gmail",
        "description": "Canonical Scaffolde Gmail capability for PAI account email.",
        "triggers": ["pai gmail", "scaffolde gmail", "email sarah"],
        "entrypoint": {
            "command": "bun",
            "args": [runner_arg],
            "cwd": "${HERMES_HOME}",
        },
        "environment": {
            "inherit": ["HOME", "PATH"],
            "set": {
                "LIFEOS_DIR": "${HERMES_HOME}",
                "GOOGLE_KEYCHAIN_ACCOUNT": "automation@scaffolde.test",
                "SCAFFOLDE_AUTOMATION_SENDER_EMAIL": "automation@scaffolde.test",
            },
        },
        "operations": {
            "search": {
                "risk": risk,
                "argv": ["search", "--query", "{query}", "--limit", "{limit}", "--untrusted-literal", "{query}"],
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "required": False, "default": 10, "minimum": 1, "maximum": 50},
                },
            },
            "send": {
                "risk": "write",
                "argv": ["send", "--to", "{to}", "--body", "{body}"],
                "parameters": {
                    "to": {"type": "string", "required": True},
                    "body": {"type": "string", "required": True},
                },
            },
        },
    }
    if extra:
        desc.update(extra)
    return desc


def _write_registry(home: Path, descriptor: dict):
    registry = home / "scaffolde" / "capabilities.json"
    registry.write_text(
        json.dumps({"version": 1, "producer": "scaffolde", "capabilities": [descriptor]}),
        encoding="utf-8",
    )
    runner = Path(descriptor.get("entrypoint", {}).get("args", [""])[0])
    try:
        runner_relative = runner.resolve().relative_to(home.resolve()).as_posix()
    except (OSError, ValueError):
        runner_relative = runner.as_posix()
    deployment = home / ".scaffolde" / "deployment.json"
    deployment.parent.mkdir(parents=True, exist_ok=True)
    managed_files = {"scaffolde/capabilities.json": hashlib.sha256(registry.read_bytes()).hexdigest()}
    if runner.exists() and runner_relative:
        managed_files[runner_relative] = hashlib.sha256(runner.read_bytes()).hexdigest()
    deployment.write_text(
        json.dumps({"surface": "hermes", "managed_files": managed_files}),
        encoding="utf-8",
    )


def _call_tool(**kwargs):
    from tools.scaffolde_capability_tool import handle_scaffolde_capability

    return json.loads(handle_scaffolde_capability(kwargs))


def test_loader_reports_absent_valid_malformed_and_degraded_status(hermes_home):
    from tools.scaffolde_capabilities import load_capability_registry

    absent = load_capability_registry()
    assert absent.status == "absent"
    assert absent.capabilities == {}

    _write_registry(hermes_home, _descriptor(hermes_home))
    valid = load_capability_registry()
    assert valid.status == "valid"
    assert list(valid.capabilities) == ["scaffolde.gmail.pai"]

    (hermes_home / "scaffolde" / "capabilities.json").write_text("{not-json", encoding="utf-8")
    malformed = load_capability_registry()
    assert malformed.status == "malformed"
    assert "registry_json" in malformed.errors[0]["code"]

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"id": ""}))
    degraded = load_capability_registry()
    assert degraded.status == "degraded"
    assert degraded.capabilities == {}
    assert degraded.errors[0]["code"] == "capability_schema"


def test_schema_path_and_secret_rejection_fail_closed(hermes_home):
    from tools.scaffolde_capabilities import load_capability_registry

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"entrypoint": {"command": "bun", "args": ["/tmp/evil.ts"], "cwd": "${HERMES_HOME}"}}))
    assert load_capability_registry().status == "degraded"
    assert any(error["code"] == "entrypoint_outside_hermes_home" for error in load_capability_registry().errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"entrypoint": {"command": "bun", "args": ["../evil.ts"], "cwd": "${HERMES_HOME}"}}))
    assert load_capability_registry().status == "degraded"
    assert any(error["code"] == "entrypoint_outside_hermes_home" for error in load_capability_registry().errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"entrypoint": {"command": "python3", "args": [str(hermes_home / "fake_runner.ts")], "cwd": "${HERMES_HOME}"}}))
    rejected_command = load_capability_registry()
    assert rejected_command.status == "degraded"
    assert any(error["code"] == "entrypoint_command" for error in rejected_command.errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"entrypoint": {"command": "bun", "args": ["--eval"], "cwd": "${HERMES_HOME}"}}))
    rejected_eval = load_capability_registry()
    assert rejected_eval.status == "degraded"
    assert any(error["code"] == "entrypoint_args" for error in rejected_eval.errors)

    unmanaged_runner = hermes_home / "operator-owned.ts"
    unmanaged_runner.write_text("console.log('must not run')\n", encoding="utf-8")
    _write_registry(hermes_home, _descriptor(hermes_home, runner=unmanaged_runner))
    deployment = hermes_home / ".scaffolde" / "deployment.json"
    registry_hash = hashlib.sha256((hermes_home / "scaffolde" / "capabilities.json").read_bytes()).hexdigest()
    deployment.write_text(
        json.dumps({"managed_files": {"scaffolde/capabilities.json": registry_hash}}),
        encoding="utf-8",
    )
    unmanaged = load_capability_registry()
    assert unmanaged.status == "degraded"
    assert any(error["code"] == "unmanaged_entrypoint" for error in unmanaged.errors)

    malformed_templates = _descriptor(hermes_home)
    malformed_templates["operations"]["search"]["argv"] = ["search", "{{query}}", "{limit}"]
    _write_registry(hermes_home, malformed_templates)
    rejected_templates = load_capability_registry()
    assert rejected_templates.status == "degraded"
    assert any("malformed templates" in error["message"] for error in rejected_templates.errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"environment": {"inherit": ["HOME"], "set": {"API_TOKEN": "literal-token"}}}))
    assert load_capability_registry().status == "degraded"
    assert load_capability_registry().errors[0]["code"] == "descriptor_secret_value"

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"environment": {"inherit": ["HOME"], "set": {"PATH": "/tmp"}}}))
    overridden_path = load_capability_registry()
    assert overridden_path.status == "degraded"
    assert any(error["code"] == "environment_override" for error in overridden_path.errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"authority": "generic"}))
    wrong_authority = load_capability_registry()
    assert wrong_authority.status == "degraded"
    assert any(error["code"] == "capability_authority" for error in wrong_authority.errors)

    _write_registry(hermes_home, _descriptor(hermes_home, extra={"version": 2}))
    assert load_capability_registry().status == "degraded"

    (hermes_home / "scaffolde" / "capabilities.json").write_text(json.dumps({"version": 2, "producer": "scaffolde", "capabilities": []}), encoding="utf-8")
    malformed = load_capability_registry()
    assert malformed.status == "malformed"

    _write_registry(hermes_home, _descriptor(hermes_home))
    registry_path = hermes_home / "scaffolde" / "capabilities.json"
    tampered_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    tampered_payload["capabilities"][0]["operations"]["send"]["risk"] = "read"
    registry_path.write_text(json.dumps(tampered_payload), encoding="utf-8")
    tampered = load_capability_registry()
    assert tampered.status == "degraded"
    assert any(error["code"] == "unmanaged_registry" for error in tampered.errors)


def test_native_tool_visible_in_default_schema_even_without_valid_registry(hermes_home):
    import model_tools

    model_tools._clear_tool_defs_cache()
    names = {t["function"]["name"] for t in model_tools.get_tool_definitions(quiet_mode=True)}
    assert "scaffolde_capability" in names

    _write_registry(hermes_home, _descriptor(hermes_home))
    model_tools._clear_tool_defs_cache()
    names = {t["function"]["name"] for t in model_tools.get_tool_definitions(quiet_mode=True)}
    assert "scaffolde_capability" in names


def test_argument_validation_no_shell_and_env_allowlist_for_read_operation(hermes_home, monkeypatch):
    runner = _write_fake_runner(hermes_home)
    _write_registry(hermes_home, _descriptor(hermes_home, runner=runner))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/should-not-leak")
    monkeypatch.setenv("SECRET_TOKEN", "should-not-inherit")

    extra = _call_tool(action="invoke", capability_id="scaffolde.gmail.pai", operation="search", arguments={"query": "Sarah; touch /tmp/pwned", "limit": 2, "extra": "nope"})
    assert extra["status"] == "error"
    assert extra["error_type"] == "invalid_arguments"

    result = _call_tool(action="invoke", capability_id="scaffolde.gmail.pai", operation="search", arguments={"query": "Sarah; touch /tmp/pwned", "limit": 2})
    assert result["status"] == "ok"
    child = json.loads(result["stdout"])
    assert child["argv"] == ["search", "--query", "Sarah; touch /tmp/pwned", "--limit", "2", "--untrusted-literal", "Sarah; touch /tmp/pwned"]
    assert child["cwd"] == str(hermes_home)
    assert child["env"]["LIFEOS_DIR"] == str(hermes_home)
    assert child["env"]["GOOGLE_APPLICATION_CREDENTIALS"] is None
    assert "should-redact" not in result["stderr"]
    assert "SECRET_TOKEN" in result["stderr"]


@pytest.mark.parametrize("operation", ["send", "reply", "archive", "mark_read", "mark_unread", "save_attachment"])
def test_write_operations_are_approval_gated(hermes_home, monkeypatch, operation):
    desc = _descriptor(hermes_home)
    desc["operations"][operation] = {"risk": "write", "argv": [operation], "parameters": {}}
    _write_registry(hermes_home, desc)
    calls = []

    def deny(tool_name, reason, *, rule_key="", approval_callback=None):
        calls.append((tool_name, reason, rule_key))
        return {"approved": False, "message": "denied"}

    monkeypatch.setattr("tools.approval.request_tool_approval", deny)
    result = _call_tool(action="invoke", capability_id="scaffolde.gmail.pai", operation=operation, arguments={})
    assert result["status"] == "error"
    assert result["error_type"] == "approval_required"
    assert len(calls) == 1
    assert calls[0][0] == "scaffolde_capability"
    assert operation in calls[0][1]
    assert "Arguments: {}" in calls[0][1]
    assert calls[0][2].startswith(f"scaffolde:scaffolde.gmail.pai:{operation}:")
    assert len(calls[0][2].rsplit(":", 1)[1]) == 12


def test_long_write_payload_uses_exact_temporary_preview(hermes_home, monkeypatch):
    _write_registry(hermes_home, _descriptor(hermes_home))
    observed = {}

    def deny_after_inspection(_tool, reason, *, rule_key="", approval_callback=None):
        match = re.search(r'review_file=("(?:[^"\\]|\\.)*")', reason)
        assert match
        preview = Path(json.loads(match.group(1)))
        observed["path"] = preview
        observed["payload"] = json.loads(preview.read_text(encoding="utf-8"))
        observed["mode"] = preview.stat().st_mode & 0o777
        observed["reason"] = reason
        observed["rule_key"] = rule_key
        return {"approved": False, "message": "denied"}

    monkeypatch.setattr("tools.approval.request_tool_approval", deny_after_inspection)
    body = "full-body-" * 300
    result = _call_tool(
        action="invoke",
        capability_id="scaffolde.gmail.pai",
        operation="send",
        arguments={"to": "person@example.com", "body": body},
    )
    assert result["error_type"] == "approval_required"
    assert observed["payload"]["body"] == body
    assert observed["mode"] == 0o600
    assert "Exact payload SHA256=" in observed["reason"]
    assert observed["rule_key"].startswith("scaffolde:scaffolde.gmail.pai:send:")
    assert not observed["path"].exists()


def test_degraded_semantics_and_list_status(hermes_home):
    _write_registry(hermes_home, _descriptor(hermes_home, extra={"id": ""}))
    status = _call_tool(action="status")
    assert status["status"] == "capability_degraded"
    listed = _call_tool(action="list")
    assert listed["status"] == "capability_degraded"
    assert "mailbox unavailable" not in json.dumps(listed).lower()


def test_prompt_routing_block_precedes_skill_guidance(hermes_home):
    _write_registry(hermes_home, _descriptor(hermes_home))
    from agent.scaffolde_capability_prompt import build_scaffolde_capabilities_prompt

    prompt = build_scaffolde_capabilities_prompt({"scaffolde_capability", "skill_view"})
    assert "Available Scaffolde capabilities" in prompt
    assert "explicit user override > matching Scaffolde authority/account > generic skills" in prompt
    assert "scaffolde.gmail.pai" in prompt
    assert "capability_degraded" in prompt


def test_incident_regression_fake_scaffolde_gmail_succeeds_without_generic_invocation(hermes_home, monkeypatch):
    runner = hermes_home / "gmail_fake.ts"
    runner.write_text(
        "import json\nprint(json.dumps({'from': 'Sarah', 'subject': 'PAI update', 'generic_invoked': False}))\n",
        encoding="utf-8",
    )
    _write_registry(hermes_home, _descriptor(hermes_home, runner=runner))
    result = _call_tool(action="invoke", capability_id="scaffolde.gmail.pai", operation="search", arguments={"query": "Sarah", "limit": 1})
    assert result["status"] == "ok"
    assert json.loads(result["stdout"])["from"] == "Sarah"
    assert json.loads(result["stdout"])["generic_invoked"] is False


def test_runtime_failures_are_capability_degraded(hermes_home):
    from tools.scaffolde_capabilities import invoke_capability

    failing = _write_fake_runner(hermes_home, exit_code=9)
    _write_registry(hermes_home, _descriptor(hermes_home, runner=failing))
    failed = invoke_capability("scaffolde.gmail.pai", "search", {"query": "Sarah", "limit": 1})
    assert failed["status"] == "capability_degraded"
    assert failed["error_type"] == "nonzero_exit"

    slow = hermes_home / "slow.ts"
    slow.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    _write_registry(hermes_home, _descriptor(hermes_home, runner=slow))
    timed_out = invoke_capability("scaffolde.gmail.pai", "search", {"query": "Sarah", "limit": 1}, timeout=1)
    assert timed_out["status"] == "capability_degraded"
    assert timed_out["error_type"] == "timeout"
