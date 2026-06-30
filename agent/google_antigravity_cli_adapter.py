"""OpenAI-compatible text facade for Google Antigravity CLI (`agy`).

Antigravity currently exposes a local CLI process, not an HTTP chat API. This
adapter gives auxiliary Hermes tasks (compression, titles, lightweight summaries)
a minimal `.chat.completions.create()` surface by invoking `agy -p ...`.

It is intentionally text-only: no tool calls, streaming, images, or model
selection. The `agy` CLI does not currently expose a model flag.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ANTIGRAVITY_MODEL = "antigravity-cli"
DEFAULT_PRINT_TIMEOUT = "5m0s"


@dataclass(frozen=True)
class AntigravityCLIError(RuntimeError):
    """Raised when the Antigravity CLI subprocess cannot return text."""

    message: str
    returncode: int | None = None

    def __str__(self) -> str:
        if self.returncode is None:
            return self.message
        return f"{self.message} (exit {self.returncode})"


def _message_content_to_text(content: Any) -> str:
    """Best-effort text extraction from OpenAI-style message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                item_type = str(item.get("type") or "")
                if item_type in {"text", "input_text"}:
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_prompt_from_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    """Convert chat messages into the single prompt accepted by `agy -p`."""
    lines: list[str] = []
    for message in messages or []:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = _message_content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            lines.append(f"System:\n{content}")
        elif role == "assistant":
            lines.append(f"Assistant:\n{content}")
        else:
            lines.append(f"User:\n{content}")
    return "\n\n".join(lines).strip()


def _completion_response(content: str, model: str) -> Any:
    """Return a small OpenAI-compatible chat completion object."""
    return SimpleNamespace(
        id="antigravity-cli-completion",
        object="chat.completion",
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content=content),
            )
        ],
    )


class _AntigravityCompletions:
    def __init__(self, client: "GoogleAntigravityCLIClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        model = str(kwargs.get("model") or self._client.model or DEFAULT_ANTIGRAVITY_MODEL)
        prompt = build_prompt_from_messages(messages)
        if not prompt:
            raise AntigravityCLIError("Antigravity CLI request had no text prompt")
        content = self._client.complete(prompt)
        return _completion_response(content, model)


class _AntigravityChat:
    def __init__(self, client: "GoogleAntigravityCLIClient") -> None:
        self.completions = _AntigravityCompletions(client)


class GoogleAntigravityCLIClient:
    """Sync OpenAI-chat facade around `agy -p`."""

    def __init__(
        self,
        *,
        command: str = "agy",
        args: Sequence[str] | None = None,
        base_url: str = "antigravity-cli://agy",
        api_key: str = "google-antigravity-cli",
        model: str = DEFAULT_ANTIGRAVITY_MODEL,
        print_timeout: str = DEFAULT_PRINT_TIMEOUT,
    ) -> None:
        self.command = command
        self.args = list(args or [])
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.print_timeout = print_timeout
        self.chat = _AntigravityChat(self)

    def complete(self, prompt: str) -> str:
        cmd = [self.command, *self.args, "-p", prompt, "--print-timeout", self.print_timeout]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AntigravityCLIError(
                f"Antigravity CLI command not found: {self.command!r}"
            ) from exc
        except Exception as exc:
            raise AntigravityCLIError(
                f"Antigravity CLI invocation failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            detail = f": {stderr[:500]}" if stderr else ""
            raise AntigravityCLIError(
                f"Antigravity CLI returned no usable completion{detail}",
                returncode=proc.returncode,
            )
        output = (proc.stdout or "").strip()
        if not output:
            stderr = (proc.stderr or "").strip()
            detail = f" stderr={stderr[:500]!r}" if stderr else ""
            raise AntigravityCLIError(
                "Antigravity CLI exited 0 but produced no output; refusing to "
                f"treat an empty compression summary as success.{detail}",
                returncode=proc.returncode,
            )
        return output


class _AsyncAntigravityCompletions:
    def __init__(self, client: "AsyncGoogleAntigravityCLIClient") -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._client.sync_client.chat.completions.create, **kwargs)


class _AsyncAntigravityChat:
    def __init__(self, client: "AsyncGoogleAntigravityCLIClient") -> None:
        self.completions = _AsyncAntigravityCompletions(client)


class AsyncGoogleAntigravityCLIClient:
    """Async facade that runs the sync CLI adapter off the event loop."""

    def __init__(self, sync_client: GoogleAntigravityCLIClient) -> None:
        self.sync_client = sync_client
        self.command = sync_client.command
        self.args = sync_client.args
        self.base_url = sync_client.base_url
        self.api_key = sync_client.api_key
        self.model = sync_client.model
        self.chat = _AsyncAntigravityChat(self)
