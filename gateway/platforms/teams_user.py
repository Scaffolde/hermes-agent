"""First-class Microsoft Teams user adapter backed by Microsoft Graph.

This adapter is intentionally separate from the bundled Bot Framework Teams
plugin.  It uses delegated Graph permissions for a real work/school Teams user
so messages are sent as that user account.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_SCOPES = [
    "User.Read",
    "Team.ReadBasic.All",
    "Channel.ReadBasic.All",
    "ChannelMessage.Read.All",
    "ChannelMessage.Send",
    "Chat.Read",
    "Chat.ReadWrite",
]


def check_teams_user_requirements() -> bool:
    """Return True when runtime dependencies for Graph access are importable."""
    try:
        import msal  # noqa: F401
    except ImportError:
        return False
    return httpx is not None


def check_requirements() -> bool:
    return check_teams_user_requirements()


def _cfg(config: PlatformConfig, key: str, env: str, default: Any = None) -> Any:
    if config.extra and key in config.extra:
        return config.extra.get(key)
    return os.getenv(env, default)


def validate_config(config: PlatformConfig) -> bool:
    """Validate minimal public-client delegated Graph settings."""
    tenant_id = _cfg(config, "tenant_id", "TEAMS_USER_TENANT_ID")
    client_id = _cfg(config, "client_id", "TEAMS_USER_CLIENT_ID")
    return bool(tenant_id and client_id)


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"br", "p", "div", "li"} and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def text(self) -> str:
        rendered = "".join(self.parts)
        rendered = re.sub(r"\n{3,}", "\n\n", rendered)
        return html.unescape(rendered).strip()


def graph_html_to_text(content: str) -> str:
    parser = _HTMLToText()
    parser.feed(content or "")
    return parser.text()


def markdownish_to_graph_html(content: str) -> str:
    """Small safe formatter for Graph chatMessage HTML bodies."""
    escaped = html.escape(content or "")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped.replace("\n", "<br>")


class MsalDelegatedTokenProvider:
    """Acquire delegated tokens for a public client app using MSAL.

    MSAL is imported lazily so unit tests and installations that only exercise
    mocked Graph clients do not require it until real authentication is used.
    """

    def __init__(self, *, tenant_id: str, client_id: str, scopes: Iterable[str], cache_path: str | None = None):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.scopes = list(scopes)
        self.cache_path = Path(cache_path or "~/.hermes/teams-user-msal-cache.bin").expanduser()
        self._app = None
        self._cache = None

    def _load(self):
        import msal  # type: ignore

        cache = msal.SerializableTokenCache()
        if self.cache_path.exists():
            cache.deserialize(self.cache_path.read_text())
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.PublicClientApplication(self.client_id, authority=authority, token_cache=cache)
        self._app = app
        self._cache = cache
        return app, cache

    def _persist(self) -> None:
        if self._cache is not None and self._cache.has_state_changed:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(self._cache.serialize())

    async def get_token(self) -> str:
        return await asyncio.to_thread(self._get_token_sync)

    def _get_token_sync(self) -> str:
        app, _cache = self._load()
        accounts = app.get_accounts()
        result = None
        if accounts:
            result = app.acquire_token_silent(self.scopes, account=accounts[0])
        if not result:
            result = app.acquire_token_interactive(scopes=self.scopes, port=0)
        self._persist()
        token = (result or {}).get("access_token")
        if not token:
            raise RuntimeError((result or {}).get("error_description") or "MSAL did not return an access token")
        return token


class GraphClient:
    def __init__(self, token_provider: Any, *, base_url: str = GRAPH_BASE_URL):
        self.token_provider = token_provider
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        return self._client

    async def request(self, method: str, path: str, *, json: Any = None, params: Dict[str, Any] | None = None) -> Any:
        token = await self.token_provider.get_token()
        client = await self._http()
        response = await client.request(
            method,
            path,
            json=json,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class TeamsUserAdapter(BasePlatformAdapter):
    """Microsoft Teams adapter using delegated Microsoft Graph user auth."""

    MAX_MESSAGE_LENGTH = 28000

    def __init__(self, config: PlatformConfig, *, graph_client: Any | None = None, token_provider: Any | None = None):
        super().__init__(config, Platform.TEAMS_USER)
        self._tenant_id = _cfg(config, "tenant_id", "TEAMS_USER_TENANT_ID")
        self._client_id = _cfg(config, "client_id", "TEAMS_USER_CLIENT_ID")
        self._user_id = _cfg(config, "user_id", "TEAMS_USER_ID")
        self._poll_interval = float(_cfg(config, "poll_interval", "TEAMS_USER_POLL_INTERVAL", 15))
        self._channels = list(config.extra.get("channels") or []) if config.extra else []
        self._chats = list(config.extra.get("chats") or []) if config.extra else []
        self._seen_message_ids: set[str] = set()
        self._poll_task: asyncio.Task | None = None
        scopes = config.extra.get("scopes") if config.extra else None
        self._scopes = list(scopes or DEFAULT_SCOPES)
        cache_path = _cfg(config, "cache_path", "TEAMS_USER_CACHE_PATH")
        if graph_client is not None:
            self._graph = graph_client
        else:
            provider = token_provider or MsalDelegatedTokenProvider(
                tenant_id=self._tenant_id,
                client_id=self._client_id,
                scopes=self._scopes,
                cache_path=cache_path,
            )
            self._graph = GraphClient(provider)

    async def connect(self) -> bool:
        if not validate_config(self.config):
            self._set_fatal_error("config", "Teams user adapter requires tenant_id and client_id", retryable=False)
            return False
        try:
            me = await self._graph.request("GET", "/me")
            self._user_id = self._user_id or me.get("id")
            self._mark_connected()
            self._poll_task = asyncio.create_task(self._poll_loop(), name="teams-user-poll")
            return True
        except Exception as exc:
            logger.error("[TeamsUser] Connect failed: %s", exc, exc_info=True)
            self._set_fatal_error("connect", str(exc), retryable=True)
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        close = getattr(self._graph, "close", None)
        if close:
            await close()

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                for event in await self.poll_once():
                    await self.handle_message(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[TeamsUser] Poll failed: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    def _split_chat_id(self, chat_id: str) -> tuple[str, str, str | None]:
        if chat_id.startswith("chat:"):
            return "chat", chat_id[len("chat:"):], None
        if "/" in chat_id:
            team_id, channel_id = chat_id.split("/", 1)
            return "channel", team_id, channel_id
        return "chat", chat_id, None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        try:
            kind, first, second = self._split_chat_id(chat_id)
            body = {"body": {"contentType": "html", "content": markdownish_to_graph_html(content)}}
            if kind == "channel":
                path = f"/teams/{first}/channels/{second}/messages"
            else:
                path = f"/chats/{first}/messages"
            response = await self._graph.request("POST", path, json=body)
            return SendResult(success=True, message_id=response.get("id"), raw_response=response)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def poll_once(self) -> list[MessageEvent]:
        events: list[MessageEvent] = []
        for channel in self._channels:
            team_id = channel["team_id"]
            channel_id = channel["channel_id"]
            response = await self._graph.request(
                "GET",
                f"/teams/{team_id}/channels/{channel_id}/messages",
                params={"$top": int(channel.get("top", 20))},
            )
            for message in reversed(response.get("value") or []):
                event = self._message_to_event(message, chat_id=f"{team_id}/{channel_id}", chat_name=channel.get("name"), chat_type="channel")
                if event:
                    events.append(event)
        for chat in self._chats:
            chat_id = chat["chat_id"]
            response = await self._graph.request("GET", f"/chats/{chat_id}/messages", params={"$top": int(chat.get("top", 20))})
            for message in reversed(response.get("value") or []):
                event = self._message_to_event(message, chat_id=f"chat:{chat_id}", chat_name=chat.get("name"), chat_type="dm")
                if event:
                    events.append(event)
        return events

    def _message_to_event(self, message: Dict[str, Any], *, chat_id: str, chat_name: str | None, chat_type: str) -> MessageEvent | None:
        message_id = str(message.get("id") or "")
        if not message_id or message_id in self._seen_message_ids:
            return None
        self._seen_message_ids.add(message_id)
        user = ((message.get("from") or {}).get("user") or {})
        user_id = user.get("id")
        if self._user_id and user_id == self._user_id:
            return None
        text = graph_html_to_text(((message.get("body") or {}).get("content") or ""))
        if not text:
            return None
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            raw_message=message,
            message_id=message_id,
            source=SessionSource(
                platform=Platform.TEAMS_USER,
                chat_id=chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user.get("displayName"),
                message_id=message_id,
            ),
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        kind, first, second = self._split_chat_id(chat_id)
        if kind == "channel":
            response = await self._graph.request("GET", f"/teams/{first}/channels/{second}")
            return {"id": chat_id, "name": response.get("displayName") or second, "type": "channel", "raw": response}
        response = await self._graph.request("GET", f"/chats/{first}")
        return {"id": chat_id, "name": response.get("topic") or first, "type": "dm", "raw": response}
