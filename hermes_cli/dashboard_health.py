"""Secret-safe health probes for the Hermes web dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib import request


_SESSION_RE = re.compile(r'__HERMES_SESSION_TOKEN__\s*=\s*["\']([^"\']+)["\']')


@dataclass
class DashboardProbeResult:
    ok: bool
    public_status_ok: bool = False
    index_ok: bool = False
    protected_api_ok: bool = False
    error: str = ""


def _read_url(url: str, *, timeout: float = 5, headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = request.Request(url, headers=headers or {})
    with request.urlopen(req, timeout=timeout) as resp:
        return int(getattr(resp, "status", 200)), resp.read().decode("utf-8", errors="replace")


def probe_dashboard(host: str = "127.0.0.1", port: int = 9119, *, timeout: float = 5) -> DashboardProbeResult:
    """Probe public and protected dashboard endpoints without printing secrets.

    Sequence mirrors actual browser access:
    1. public /api/status
    2. index page containing ephemeral session token
    3. protected /api/logs with X-Hermes-Session-Token
    """
    base = f"http://{host}:{port}"
    try:
        status_code, status_body = _read_url(f"{base}/api/status", timeout=timeout)
        public_ok = status_code == 200
        if public_ok:
            try:
                json.loads(status_body or "{}")
            except Exception:
                public_ok = False

        index_code, html = _read_url(f"{base}/", timeout=timeout)
        token_match = _SESSION_RE.search(html)
        index_ok = index_code == 200 and bool(token_match)
        if not token_match:
            return DashboardProbeResult(
                ok=False,
                public_status_ok=public_ok,
                index_ok=index_ok,
                protected_api_ok=False,
                error="dashboard index did not expose a session token",
            )

        token = token_match.group(1)
        protected_code, _ = _read_url(
            f"{base}/api/logs?file=errors&lines=1",
            timeout=timeout,
            headers={"X-Hermes-Session-Token": token},
        )
        protected_ok = protected_code == 200
        return DashboardProbeResult(
            ok=public_ok and index_ok and protected_ok,
            public_status_ok=public_ok,
            index_ok=index_ok,
            protected_api_ok=protected_ok,
        )
    except Exception as exc:
        return DashboardProbeResult(ok=False, error=str(exc))


def format_probe_result(result: DashboardProbeResult) -> str:
    state = "healthy" if result.ok else "unhealthy"
    lines = [f"Dashboard health: {state}"]
    lines.append(f"  public /api/status: {'ok' if result.public_status_ok else 'fail'}")
    lines.append(f"  index token: {'ok' if result.index_ok else 'fail'}")
    lines.append(f"  protected API: {'ok' if result.protected_api_ok else 'fail'}")
    if result.error:
        lines.append(f"  error: {result.error}")
    return "\n".join(lines)
