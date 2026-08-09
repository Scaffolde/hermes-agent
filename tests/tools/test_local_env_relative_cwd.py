"""Regression tests for local terminal initial cwd normalization."""

import os
from pathlib import Path

import pytest

from tools.environments import local
from tools.environments.local import LocalEnvironment, _resolve_local_initial_cwd


def test_relative_initial_cwd_resolves_from_parent(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    assert _resolve_local_initial_cwd("hermes-agent") == str(project)


def test_local_environment_keeps_existing_relative_child_cwd(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    env = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = env.execute("pwd", timeout=5)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip() == str(project)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership and symlink checks")
class TestInstalledUserRuntimePath:
    def test_accepts_owned_regular_bun(self, tmp_path, monkeypatch):
        bun_dir = tmp_path / ".bun" / "bin"
        bun_dir.mkdir(parents=True)
        bun = bun_dir / "bun"
        bun.write_text("#!/bin/sh\n")
        bun.chmod(0o700)
        monkeypatch.setattr(local.os.path, "expanduser", lambda value: str(tmp_path))

        assert local._installed_user_runtime_path_entries(str(tmp_path)) == [
            str(bun_dir)
        ]

    def test_rejects_symlinked_bun(self, tmp_path, monkeypatch):
        bun_dir = tmp_path / ".bun" / "bin"
        bun_dir.mkdir(parents=True)
        target = tmp_path / "bun-real"
        target.write_text("#!/bin/sh\n")
        target.chmod(0o700)
        (bun_dir / "bun").symlink_to(target)
        monkeypatch.setattr(local.os.path, "expanduser", lambda value: str(tmp_path))

        assert local._installed_user_runtime_path_entries(str(tmp_path)) == []

    def test_rejects_non_executable_bun(self, tmp_path, monkeypatch):
        bun_dir = tmp_path / ".bun" / "bin"
        bun_dir.mkdir(parents=True)
        bun = bun_dir / "bun"
        bun.write_text("#!/bin/sh\n")
        bun.chmod(0o600)
        monkeypatch.setattr(local.os.path, "expanduser", lambda value: str(tmp_path))

        assert local._installed_user_runtime_path_entries(str(tmp_path)) == []

    def test_rejects_noncanonical_home(self, tmp_path):
        foreign_home = tmp_path / "foreign"
        bun_dir = foreign_home / ".bun" / "bin"
        bun_dir.mkdir(parents=True)
        bun = bun_dir / "bun"
        bun.write_text("#!/bin/sh\n")
        bun.chmod(0o700)

        assert local._installed_user_runtime_path_entries(str(foreign_home)) == []
