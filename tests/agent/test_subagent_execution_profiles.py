"""Focused tests for host-resolved subagent execution profiles.

Profiles are declared in ``config.yaml`` under
``delegation.execution_profiles.<id>`` and resolved fail-closed by
``agent.subagent_execution_profiles``.  These tests exercise the real
resolver against a temp ``HERMES_HOME`` (the autouse conftest fixture
redirects the env var), including one end-to-end path through the real
config loader with no injected config dict.
"""

import dataclasses
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import agent.subagent_execution_profiles as execution_profiles

from agent.subagent_execution_profiles import (
    ExecutionProfileError,
    ResolvedExecutionProfile,
    _read_regular_file_beneath,
    check_profile_transition,
    resolve_execution_profile,
    validate_profile_id,
)
from hermes_constants import get_hermes_home

PROTOCOL_TEXT = "# Reviewer protocol\nReview the diff. Report findings only.\n"


@pytest.fixture
def hermes_home() -> Path:
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "protocols").mkdir(exist_ok=True)
    (home / "protocols" / "reviewer.md").write_text(PROTOCOL_TEXT, encoding="utf-8")
    return home


def _profile_config(**overrides):
    profile = {
        "protocol_file": "protocols/reviewer.md",
        "role": "leaf",
        "allowed_toolsets": ["file"],
        "expected_tool_names": ["read_file", "write_file"],
        "allow_root": True,
    }
    profile.update(overrides)
    return {"delegation": {"execution_profiles": {"reviewer": profile}}}


def test_valid_profile_resolves_with_pinned_protocol_hash(hermes_home):
    profile = resolve_execution_profile("reviewer", config=_profile_config())
    assert profile.profile_id == "reviewer"
    assert profile.role == "leaf"
    assert profile.allowed_toolsets == ("file",)
    assert profile.expected_tool_names == frozenset({"read_file", "write_file"})
    assert profile.protocol_text == PROTOCOL_TEXT
    assert (
        profile.protocol_sha256
        == hashlib.sha256(PROTOCOL_TEXT.encode("utf-8")).hexdigest()
    )
    assert profile.allow_root is True
    assert profile.allowed_child_profiles == ()
    assert profile.timeout_seconds is None
    assert profile.execution_backend == "in_process"
    assert profile.workspace_root is None
    assert profile.cgroup_parent is None
    assert profile.max_process_iterations == 8


def test_process_profile_requires_canonical_non_root_workspace(hermes_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = resolve_execution_profile(
        "reviewer",
        config=_profile_config(
            execution_backend="portable",
            workspace_root=str(workspace),
            max_process_iterations=3,
        ),
    )
    assert profile.execution_backend == "portable"
    assert profile.workspace_root == str(workspace.resolve())
    assert profile.max_process_iterations == 3

    bounded = resolve_execution_profile(
        "reviewer",
        config=_profile_config(
            execution_backend="portable",
            workspace_root=str(workspace),
            max_process_iterations=64,
        ),
    )
    assert bounded.max_process_iterations == 64

    with pytest.raises(ExecutionProfileError, match=r"\[1, 64\]"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(
                execution_backend="portable",
                workspace_root=str(workspace),
                max_process_iterations=65,
            ),
        )

    with pytest.raises(ExecutionProfileError, match="require an absolute"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(execution_backend="portable", workspace_root="relative"),
        )
    with pytest.raises(ExecutionProfileError, match="non-root"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(execution_backend="portable", workspace_root="/"),
        )


def test_linux_strict_profile_requires_host_delegated_cgroup_parent(
    hermes_home, tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ExecutionProfileError, match="cgroup_parent"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(
                execution_backend="linux_strict",
                workspace_root=str(workspace),
            ),
        )

    delegated = tmp_path / "delegated-cgroup"
    delegated.mkdir()
    profile = resolve_execution_profile(
        "reviewer",
        config=_profile_config(
            execution_backend="linux_strict",
            workspace_root=str(workspace),
            cgroup_parent=str(delegated),
        ),
    )
    assert profile.cgroup_parent == str(delegated.resolve())


def test_portable_profile_rejects_cgroup_parent(hermes_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    delegated = tmp_path / "delegated-cgroup"
    delegated.mkdir()
    with pytest.raises(ExecutionProfileError, match="only valid for linux_strict"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(
                execution_backend="portable",
                workspace_root=str(workspace),
                cgroup_parent=str(delegated),
            ),
        )


def test_profile_resolves_host_owned_blocked_tools(hermes_home):
    profile = resolve_execution_profile(
        "reviewer",
        config=_profile_config(blocked_tools=["patch", "read_terminal"]),
    )

    assert profile.blocked_tools == frozenset({"patch", "read_terminal"})
    assert profile.expected_tool_names.isdisjoint(profile.blocked_tools)


def test_resolved_profile_is_immutable(hermes_home):
    profile = resolve_execution_profile("reviewer", config=_profile_config())
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.role = "orchestrator"
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.expected_tool_names = frozenset()


def test_profile_resolution_uses_invocation_config_snapshot(hermes_home, monkeypatch):
    config = _profile_config()

    def mutate_after_initial_reads():
        config["delegation"]["execution_profiles"]["reviewer"][
            "expected_tool_names"
        ] = ["terminal"]
        return {"file"}

    monkeypatch.setattr(
        execution_profiles, "_known_toolset_names", mutate_after_initial_reads
    )

    profile = resolve_execution_profile("reviewer", config=config)

    assert profile.expected_tool_names == frozenset({"read_file", "write_file"})


@pytest.mark.parametrize(
    "unsafe",
    [
        "clarify",
        "delegate_task",
        "execute_code",
        "memory",
        "read_terminal",
        "session_search",
        "todo",
        "tool_search",
        "tool_describe",
        "tool_call",
    ],
)
def test_profile_rejects_tools_without_frozen_adapters(hermes_home, unsafe):
    with pytest.raises(ExecutionProfileError, match="without frozen profile adapters"):
        resolve_execution_profile(
            "reviewer",
            config=_profile_config(expected_tool_names=[unsafe]),
        )


@pytest.mark.parametrize(
    "bad_id",
    ["", "UPPER", "has space", "a/b", "../x", "-lead", "a" * 65, None, 7],
)
def test_profile_id_validation_rejects_unsafe_identifiers(bad_id):
    with pytest.raises(ExecutionProfileError):
        validate_profile_id(bad_id)


def test_unknown_profile_fails_closed(hermes_home):
    with pytest.raises(ExecutionProfileError, match="Unknown execution profile"):
        resolve_execution_profile("reviewer", config={"delegation": {}})
    with pytest.raises(ExecutionProfileError, match="Unknown execution profile"):
        resolve_execution_profile("other", config=_profile_config())


@pytest.mark.parametrize(
    "protocol_file,match",
    [
        ("/etc/passwd", "absolute"),
        ("protocols/../../outside.md", "'.' or '..'"),
        ("./protocols/reviewer.md", "'.' or '..'"),
        ("protocols/missing.md", "unsafe"),
        ("protocols", "regular file"),
        ("", "non-empty"),
        (None, "non-empty"),
        ("protocols/rev\x00.md", "NUL"),
    ],
)
def test_protocol_path_safety_rejections(hermes_home, protocol_file, match):
    config = _profile_config(protocol_file=protocol_file)
    with pytest.raises(ExecutionProfileError, match=match):
        resolve_execution_profile("reviewer", config=config)


def test_protocol_symlink_escape_rejected(hermes_home, tmp_path):
    outside = tmp_path / "outside-protocol.md"
    outside.write_text("outside", encoding="utf-8")
    link = hermes_home / "protocols" / "escape.md"
    os.symlink(outside, link)
    config = _profile_config(protocol_file="protocols/escape.md")
    with pytest.raises(ExecutionProfileError, match="unsafe"):
        resolve_execution_profile("reviewer", config=config)


def test_protocol_symlink_inside_home_is_rejected(hermes_home):
    link = hermes_home / "protocols" / "alias.md"
    os.symlink(hermes_home / "protocols" / "reviewer.md", link)
    config = _profile_config(protocol_file="protocols/alias.md")
    with pytest.raises(ExecutionProfileError, match="unsafe"):
        resolve_execution_profile("reviewer", config=config)


def test_protocol_loading_fails_closed_without_descriptor_relative_open(
    hermes_home, monkeypatch
):
    monkeypatch.setattr(execution_profiles.os, "supports_dir_fd", set())
    with pytest.raises(ExecutionProfileError, match="descriptor-relative"):
        resolve_execution_profile("reviewer", config=_profile_config())


@pytest.mark.skipif(not getattr(os, "O_NOFOLLOW", 0), reason="requires O_NOFOLLOW")
def test_protocol_root_replacement_symlink_is_rejected(tmp_path):
    root = tmp_path / "hermes-home"
    root.mkdir()
    target = root / "protocol.md"
    target.write_text("trusted", encoding="utf-8")

    original = tmp_path / "original-home"
    root.rename(original)
    attacker = tmp_path / "attacker-home"
    attacker.mkdir()
    (attacker / "protocol.md").write_text("attacker", encoding="utf-8")
    os.symlink(attacker, root, target_is_directory=True)

    with pytest.raises(ExecutionProfileError, match="changed or became unsafe"):
        _read_regular_file_beneath(root, target)


@pytest.mark.skipif(
    os.open not in getattr(os, "supports_dir_fd", set()),
    reason="requires descriptor-relative open",
)
def test_protocol_read_uses_already_opened_root_after_directory_replacement(tmp_path):
    root = tmp_path / "hermes-home"
    root.mkdir()
    target = root / "protocol.md"
    target.write_text("trusted", encoding="utf-8")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        original = tmp_path / "original-home"
        root.rename(original)
        root.mkdir()
        (root / "protocol.md").write_text("attacker", encoding="utf-8")

        assert (
            _read_regular_file_beneath(root, target, root_fd=root_fd) == b"trusted"
        )
    finally:
        os.close(root_fd)


@pytest.mark.skipif(
    os.open not in getattr(os, "supports_dir_fd", set()),
    reason="requires descriptor-relative open",
)
def test_profile_resolution_holds_original_root_across_directory_swap(
    hermes_home, monkeypatch
):
    original_open = execution_profiles._open_verified_root
    moved_home = hermes_home.with_name("original-hermes-home")

    def open_then_replace(root):
        fd = original_open(root)
        hermes_home.rename(moved_home)
        (hermes_home / "protocols").mkdir(parents=True)
        (hermes_home / "protocols" / "reviewer.md").write_text(
            "attacker", encoding="utf-8"
        )
        return fd

    monkeypatch.setattr(execution_profiles, "_open_verified_root", open_then_replace)

    profile = resolve_execution_profile("reviewer", config=_profile_config())

    assert profile.protocol_text == PROTOCOL_TEXT


def test_protocol_oversized_rejected(hermes_home):
    big = hermes_home / "protocols" / "big.md"
    big.write_bytes(b"x" * (32_000 + 1))
    config = _profile_config(protocol_file="protocols/big.md")
    with pytest.raises(ExecutionProfileError, match="exceeds"):
        resolve_execution_profile("reviewer", config=config)


def test_protocol_non_utf8_rejected(hermes_home):
    binary = hermes_home / "protocols" / "binary.md"
    binary.write_bytes(b"\xff\xfe\x00bad")
    config = _profile_config(protocol_file="protocols/binary.md")
    with pytest.raises(ExecutionProfileError, match="UTF-8"):
        resolve_execution_profile("reviewer", config=config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("allowed_toolsets", []),
        ("allowed_toolsets", None),
        ("allowed_toolsets", ["file", ""]),
        ("allowed_toolsets", ["file", "file"]),
        ("allowed_toolsets", "file"),
        ("expected_tool_names", []),
        ("expected_tool_names", None),
        ("expected_tool_names", ["read_file", "read_file"]),
        ("role", "root"),
        ("allow_root", "yes"),
        ("allowed_child_profiles", ["Bad/Id"]),
        ("timeout_seconds", True),
        ("timeout_seconds", -1),
        ("timeout_seconds", 0),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", 86_401),
    ],
)
def test_malformed_profile_shapes_fail_closed(hermes_home, field, value):
    with pytest.raises(ExecutionProfileError):
        resolve_execution_profile("reviewer", config=_profile_config(**{field: value}))


def test_unknown_profile_keys_fail_closed(hermes_home):
    with pytest.raises(ExecutionProfileError, match="Unknown keys"):
        resolve_execution_profile(
            "reviewer", config=_profile_config(typo_allowed_tools=["read_file"])
        )


def test_unknown_toolset_rejected_but_registry_toolsets_accepted(
    hermes_home, monkeypatch
):
    with pytest.raises(ExecutionProfileError, match="Unknown toolsets"):
        resolve_execution_profile(
            "reviewer", config=_profile_config(allowed_toolsets=["definitely-nope"])
        )
    # Plugin/MCP toolsets are recognized through the live registry, not just
    # the static TOOLSETS dict.
    from tools.registry import registry

    real = registry.get_available_toolsets

    def with_plugin_toolset():
        toolsets = dict(real())
        toolsets["plugin-extra"] = {"available": True, "tools": ["plugin_tool"]}
        return toolsets

    monkeypatch.setattr(registry, "get_available_toolsets", with_plugin_toolset)
    profile = resolve_execution_profile(
        "reviewer", config=_profile_config(allowed_toolsets=["plugin-extra"])
    )
    assert profile.allowed_toolsets == ("plugin-extra",)


def test_timeout_and_child_profiles_resolve(hermes_home):
    profile = resolve_execution_profile(
        "reviewer",
        config=_profile_config(timeout_seconds=90, allowed_child_profiles=["verifier"]),
    )
    assert profile.timeout_seconds == 90.0
    assert profile.allowed_child_profiles == ("verifier",)


# ── Transition graph (host-observed parent attrs only) ─────────────────────


def _resolved(profile_id="reviewer", **overrides):
    fields = dict(
        profile_id=profile_id,
        role="leaf",
        allowed_toolsets=("file",),
        expected_tool_names=frozenset({"read_file"}),
        protocol_file="protocols/reviewer.md",
        protocol_text="p",
        protocol_sha256="0" * 64,
        allow_root=False,
        allowed_child_profiles=(),
        timeout_seconds=None,
    )
    fields.update(overrides)
    return ResolvedExecutionProfile(**fields)


def test_root_launch_requires_allow_root():
    parent = SimpleNamespace()  # host never stamped a profile → root launch
    with pytest.raises(ExecutionProfileError, match="allow_root"):
        check_profile_transition(parent, _resolved(allow_root=False))
    check_profile_transition(parent, _resolved(allow_root=True))


def test_profile_parent_may_only_launch_listed_children():
    candidate = _resolved("candidate", allowed_child_profiles=("verifier",))
    parent = SimpleNamespace(
        _execution_profile_id="candidate", _execution_profile=candidate
    )
    check_profile_transition(parent, _resolved("verifier"))
    with pytest.raises(ExecutionProfileError, match="allowed_child_profiles"):
        check_profile_transition(parent, _resolved("reviewer"))


def test_inconsistent_parent_profile_state_fails_closed():
    parent = SimpleNamespace(_execution_profile_id="candidate")
    with pytest.raises(ExecutionProfileError, match="inconsistent"):
        check_profile_transition(parent, _resolved("verifier", allow_root=True))
    parent = SimpleNamespace(
        _execution_profile_id="candidate",
        _execution_profile=_resolved("other-id"),
    )
    with pytest.raises(ExecutionProfileError, match="inconsistent"):
        check_profile_transition(parent, _resolved("verifier", allow_root=True))


def test_model_metadata_never_grants_transitions():
    """The graph keys off host-set attrs, never model-writable metadata."""
    parent = SimpleNamespace(
        metadata={"_execution_profile_id": "candidate"},
    )
    # No host-stamped attr → treated as root, so allow_root governs.
    with pytest.raises(ExecutionProfileError, match="allow_root"):
        check_profile_transition(parent, _resolved(allow_root=False))


# ── Real-import E2E: config.yaml on disk, real loader, no injected config ──


def test_e2e_resolution_through_real_config_loader(hermes_home):
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(_profile_config(timeout_seconds=30)), encoding="utf-8"
    )
    profile = resolve_execution_profile("reviewer")
    assert (
        profile.protocol_sha256
        == hashlib.sha256(
            (hermes_home / "protocols" / "reviewer.md").read_bytes()
        ).hexdigest()
    )
    assert profile.timeout_seconds == 30.0
    # Fail-closed against the same real loader: undeclared id.
    with pytest.raises(ExecutionProfileError, match="Unknown execution profile"):
        resolve_execution_profile("not-declared")
