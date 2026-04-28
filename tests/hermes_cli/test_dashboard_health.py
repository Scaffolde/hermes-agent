from __future__ import annotations

import io
from urllib.error import HTTPError


def test_dashboard_health_probe_fetches_public_and_protected_endpoint(monkeypatch):
    from hermes_cli.dashboard_health import probe_dashboard

    calls = []

    class Resp:
        def __init__(self, body: str, status: int = 200):
            self.status = status
            self._body = body.encode()
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=5):
        url = getattr(req, "full_url", req)
        headers = dict(getattr(req, "headers", {}) or {})
        calls.append((url, headers))
        if url.endswith("/api/status"):
            return Resp('{"ok": true}')
        if url.endswith("/"):
            return Resp('<script>window.__HERMES_SESSION_TOKEN__="tok123"</script>')
        if url.endswith("/api/logs?file=errors&lines=1"):
            assert headers.get("X-hermes-session-token") == "tok123" or headers.get("X-Hermes-Session-Token") == "tok123"
            return Resp('{"lines": []}')
        raise AssertionError(url)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_dashboard("127.0.0.1", 9119)
    assert result.ok is True
    assert result.public_status_ok is True
    assert result.index_ok is True
    assert result.protected_api_ok is True
    assert len(calls) == 3


def test_dashboard_health_probe_reports_down_without_traceback(monkeypatch):
    from hermes_cli.dashboard_health import probe_dashboard

    def fake_urlopen(req, timeout=5):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_dashboard("127.0.0.1", 9119)
    assert result.ok is False
    assert "connection refused" in result.error
