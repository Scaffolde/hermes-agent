"""Tests for the first-class Microsoft Teams Graph user adapter."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.teams_user import TeamsUserAdapter, check_teams_user_requirements, validate_config


class FakeGraphClient:
    def __init__(self):
        self.requests = []
        self.responses = []

    def queue(self, response):
        self.responses.append(response)

    async def request(self, method, path, *, json=None, params=None):
        self.requests.append({"method": method, "path": path, "json": json, "params": params})
        if not self.responses:
            return {}
        return self.responses.pop(0)


def _config(**extra):
    return PlatformConfig(enabled=True, extra=extra)


def test_validate_config_accepts_public_client_delegated_graph_settings(monkeypatch):
    monkeypatch.delenv("TEAMS_USER_TENANT_ID", raising=False)
    monkeypatch.delenv("TEAMS_USER_CLIENT_ID", raising=False)
    cfg = _config(tenant_id="tenant", client_id="client")

    assert validate_config(cfg) is True


def test_validate_config_rejects_missing_tenant_or_client(monkeypatch):
    monkeypatch.delenv("TEAMS_USER_TENANT_ID", raising=False)
    monkeypatch.delenv("TEAMS_USER_CLIENT_ID", raising=False)

    assert validate_config(_config(tenant_id="tenant")) is False
    assert validate_config(_config(client_id="client")) is False


def test_requirements_fail_when_msal_is_not_importable(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "msal":
            raise ImportError("no msal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert check_teams_user_requirements() is False


def test_gateway_config_treats_teams_user_extra_credentials_as_connected():
    cfg = GatewayConfig(
        platforms={
            Platform.TEAMS_USER: _config(tenant_id="tenant", client_id="client"),
        }
    )

    assert cfg.get_connected_platforms() == [Platform.TEAMS_USER]


@pytest.mark.asyncio
async def test_send_to_channel_uses_graph_channel_message_endpoint():
    graph = FakeGraphClient()
    graph.queue({"id": "msg-123"})
    adapter = TeamsUserAdapter(_config(tenant_id="tenant", client_id="client"), graph_client=graph)

    result = await adapter.send("team-1/channel-1", "hello **teams**")

    assert result.success is True
    assert result.message_id == "msg-123"
    assert graph.requests == [
        {
            "method": "POST",
            "path": "/teams/team-1/channels/channel-1/messages",
            "json": {"body": {"contentType": "html", "content": "hello <strong>teams</strong>"}},
            "params": None,
        }
    ]


@pytest.mark.asyncio
async def test_send_to_chat_uses_graph_chat_message_endpoint():
    graph = FakeGraphClient()
    graph.queue({"id": "chat-msg-1"})
    adapter = TeamsUserAdapter(_config(tenant_id="tenant", client_id="client"), graph_client=graph)

    result = await adapter.send("chat:19:abc@thread.v2", "hello")

    assert result.success is True
    assert result.message_id == "chat-msg-1"
    assert graph.requests[0]["method"] == "POST"
    assert graph.requests[0]["path"] == "/chats/19:abc@thread.v2/messages"


@pytest.mark.asyncio
async def test_poll_channel_messages_emits_non_self_messages_once():
    graph = FakeGraphClient()
    graph.queue(
        {
            "value": [
                {
                    "id": "old-self",
                    "createdDateTime": "2026-05-07T20:00:00Z",
                    "from": {"user": {"id": "me", "displayName": "Pai Scaffolde"}},
                    "body": {"content": "self"},
                },
                {
                    "id": "new-1",
                    "createdDateTime": "2026-05-07T20:00:01Z",
                    "from": {"user": {"id": "gary", "displayName": "Gary"}},
                    "body": {"content": "<p>Hello <b>Pai</b></p>"},
                },
            ]
        }
    )
    adapter = TeamsUserAdapter(
        _config(
            tenant_id="tenant",
            client_id="client",
            user_id="me",
            channels=[{"team_id": "team-1", "channel_id": "channel-1", "name": "General"}],
        ),
        graph_client=graph,
    )

    events = await adapter.poll_once()
    events_again = await adapter.poll_once()

    assert len(events) == 1
    event = events[0]
    assert event.message_id == "new-1"
    assert event.message_type == MessageType.TEXT
    assert event.text == "Hello Pai"
    assert event.source.platform == Platform.TEAMS_USER
    assert event.source.chat_id == "team-1/channel-1"
    assert event.source.chat_name == "General"
    assert event.source.chat_type == "channel"
    assert event.source.user_id == "gary"
    assert event.source.user_name == "Gary"
    assert events_again == []
