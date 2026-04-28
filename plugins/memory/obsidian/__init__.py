"""Obsidian/SecondBrain filesystem-backed memory provider.

This provider treats an Obsidian vault as a human-readable durable archive for
Hermes memory. It intentionally writes Markdown directly so it works even when
Obsidian is closed and no Obsidian CLI or REST plugin is installed.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent.memory_provider import MemoryProvider


def _load_config() -> dict:
    try:
        from hermes_constants import get_hermes_home

        cfg_path = get_hermes_home() / "config.yaml"
        if not cfg_path.exists():
            return {}
        all_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mem = all_cfg.get("memory", {}) or {}
        return mem.get("obsidian", {}) or {}
    except Exception:
        return {}


def _default_vault() -> Path:
    for candidate in (
        Path.home() / "SecondBrain",
        Path.home() / "Documents" / "Obsidian Vault",
    ):
        if candidate.exists():
            return candidate
    return Path.home() / "SecondBrain"


def _slug(text: str, *, fallback: str = "memory") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (text[:72].strip("-") or fallback)


def _keywords(text: str) -> set[str]:
    stop = {"the", "and", "for", "that", "with", "this", "what", "when", "where", "prefer", "prefers", "memory"}
    return {w for w in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()) if w not in stop}


class ObsidianMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._vault_path: Path = Path(self._config.get("vault_path") or _default_vault()).expanduser()
        self._folder = str(self._config.get("folder") or "AI Memory/Hermes").strip("/")
        self._root: Path = self._vault_path / self._folder
        self._session_id = ""

    @property
    def name(self) -> str:
        return "obsidian"

    def is_available(self) -> bool:
        try:
            self._vault_path.mkdir(parents=True, exist_ok=True)
            (self._vault_path / self._folder).mkdir(parents=True, exist_ok=True)
            return self._vault_path.is_dir() and (self._vault_path / self._folder).is_dir()
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "vault_path", "description": "Obsidian vault path", "default": str(_default_vault())},
            {"key": "folder", "description": "Folder inside vault", "default": "AI Memory/Hermes"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        cfg_path = Path(hermes_home) / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg = cfg or {}
        cfg.setdefault("memory", {})
        cfg["memory"]["provider"] = "obsidian"
        cfg["memory"]["obsidian"] = values
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._vault_path.mkdir(parents=True, exist_ok=True)
        self._root.mkdir(parents=True, exist_ok=True)

    def system_prompt_block(self) -> str:
        return (
            "# Obsidian Memory\n"
            f"Active. Vault: {self._vault_path}. Folder: {self._folder}.\n"
            "Stores durable Markdown memory notes for human-auditable recall."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def _write_note(self, title: str, body: str, *, scope: str = "fact", source: str = "hermes") -> Path:
        now = datetime.now(timezone.utc).isoformat()
        filename = f"{now[:10]}-{_slug(title)}.md"
        path = self._root / filename
        i = 2
        while path.exists():
            path = self._root / f"{now[:10]}-{_slug(title)}-{i}.md"
            i += 1
        frontmatter = {
            "type": "hermes-memory",
            "scope": scope,
            "source": source,
            "confidence": "high",
            "created": now,
            "updated": now,
            "tags": ["hermes-memory"],
        }
        content = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False).strip() + "\n---\n\n"
        content += f"# {title}\n\n{body.strip()}\n"
        path.write_text(content, encoding="utf-8")
        return path

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace"} or not content:
            return
        scope = "user" if target == "user" else "environment"
        self._write_note(content[:80], content, scope=scope, source="hermes-memory-tool")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not user_content and not assistant_content:
            return
        body = f"## User\n\n{user_content.strip()}\n\n## Assistant\n\n{assistant_content.strip()}"
        self._write_note(f"Session turn {session_id or self._session_id or 'unknown'}", body, scope="episode")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        excerpt = "\n".join(
            f"- {m.get('role','unknown')}: {str(m.get('content',''))[:500]}" for m in messages[-12:]
        )
        self._write_note("Pre-compression snapshot", excerpt, scope="episode")
        return "Obsidian memory archived a pre-compression Markdown snapshot."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        q = _keywords(query)
        if not q or not self._root.exists():
            return ""
        scored: list[tuple[int, Path, str]] = []
        for path in sorted(self._root.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:500]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            score = len(q & _keywords(text))
            if score:
                snippet = " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("---"))[:500]
                scored.append((score, path, snippet))
        if not scored:
            return ""
        scored.sort(key=lambda item: (-item[0], item[1].name))
        lines = [f"- [[{p.relative_to(self._vault_path).with_suffix('')}]]: {snippet}" for _, p, snippet in scored[:5]]
        return "## Obsidian Memory\n" + "\n".join(lines)


def register(ctx):
    ctx.register_memory_provider(ObsidianMemoryProvider())
