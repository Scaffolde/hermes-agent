from __future__ import annotations

import json
from pathlib import Path


def test_obsidian_provider_discovers_and_uses_configured_vault(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    vault = tmp_path / "SecondBrain"
    hermes_home.mkdir()
    vault.mkdir()
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  provider: obsidian\n"
        "  obsidian:\n"
        f"    vault_path: {vault}\n"
        "    folder: AI Memory/Hermes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins.memory import discover_memory_providers, load_memory_provider

    assert "obsidian" in {name for name, _, _ in discover_memory_providers()}
    provider = load_memory_provider("obsidian")
    assert provider is not None
    assert provider.name == "obsidian"
    assert provider.is_available() is True

    provider.initialize("session-1", hermes_home=str(hermes_home), platform="cli")
    assert (vault / "AI Memory/Hermes").is_dir()
    assert "Obsidian Memory" in provider.system_prompt_block()


def test_obsidian_provider_mirrors_builtin_memory_write_to_markdown(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    vault = tmp_path / "SecondBrain"
    hermes_home.mkdir()
    vault.mkdir()
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  provider: obsidian\n"
        "  obsidian:\n"
        f"    vault_path: {vault}\n"
        "    folder: AI Memory/Hermes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins.memory import load_memory_provider

    provider = load_memory_provider("obsidian")
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="cli")
    provider.on_memory_write("add", "user", "Gary prefers live verification", {"session_id": "session-1"})

    notes = list((vault / "AI Memory/Hermes").glob("*.md"))
    assert len(notes) == 1
    content = notes[0].read_text(encoding="utf-8")
    assert "type: hermes-memory" in content
    assert "scope: user" in content
    assert "Gary prefers live verification" in content


def test_obsidian_provider_prefetch_keyword_recalls_markdown(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    vault = tmp_path / "SecondBrain"
    folder = vault / "AI Memory/Hermes"
    hermes_home.mkdir()
    folder.mkdir(parents=True)
    (folder / "gary-live-verification.md").write_text(
        "---\ntype: hermes-memory\nscope: user\n---\n\nGary prefers live verification over score gaming.\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "memory:\n"
        "  provider: obsidian\n"
        "  obsidian:\n"
        f"    vault_path: {vault}\n"
        "    folder: AI Memory/Hermes\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins.memory import load_memory_provider

    provider = load_memory_provider("obsidian")
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="cli")
    recalled = provider.prefetch("Do we prefer live verification?")
    assert "Obsidian Memory" in recalled
    assert "live verification" in recalled
