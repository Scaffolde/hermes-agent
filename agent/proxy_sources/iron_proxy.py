"""iron-proxy (`ironsh/iron-proxy`) integration for credential-injecting egress control.

Why
---

Remote terminal sandboxes (Docker, Modal, SSH) currently see real upstream
API credentials.  A prompt-injected agent inside one of these sandboxes can
``cat ~/.config/openrouter/auth.json`` or ``printenv | grep -i key`` and
exfiltrate them.

iron-proxy is a TLS-intercepting egress firewall (Apache-2.0, Go binary, by
ironsh).  It sits between the sandbox and the internet, enforces a default-deny
allowlist on outbound hosts, and *swaps proxy tokens for real credentials*
on the way out.  The sandbox only ever holds opaque proxy tokens — leaking
them is useless, since they only work from behind the proxy.

Design summary
--------------

* The ``iron-proxy`` binary is auto-installed into ``<hermes_home>/bin/iron-proxy``
  on first use.  Hermes pins one upstream version (``_IRON_PROXY_VERSION``)
  and downloads the matching tar.gz from the official GitHub Releases page,
  verifying the SHA-256 against the release's ``checksums.txt``.

* A long-lived CA at ``<hermes_home>/proxy/ca.{crt,key}`` is generated on
  first ``hermes proxy setup``.  Sandboxes trust this CA so iron-proxy can
  terminate TLS and rewrite headers.

* The proxy config lives at ``<hermes_home>/proxy/proxy.yaml``.  It enumerates
  the per-provider allowlists and the ``secrets`` transform that does the
  Authorization-header swap.

* Token mappings (proxy token -> real credential lookup) live alongside the
  config.  The real credential is **never** written to the config — iron-proxy
  reads it from its own environment via ``{type: env, var: NAME}``.  When
  Bitwarden Secrets Manager is configured, the real value is pulled there
  at proxy startup instead.

* The proxy runs as a managed subprocess (``hermes proxy start``), pidfile
  at ``<hermes_home>/proxy/iron-proxy.pid``, structured audit log at
  ``<hermes_home>/proxy/audit.log``.

* Failures (binary missing, port collision, bad config) emit a one-line
  warning and do *not* block agent startup.  The Docker backend refuses to
  start a sandbox with the proxy enabled-but-down, with a clear error.

This module is intentionally subprocess-driven rather than depending on any
iron-proxy Python bindings — a single cross-platform binary is easier to
lazy-install than a wheels-with-extension dependency, and we keep maintenance
to a "bump the pinned version" loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Pinned upstream version.  Bump in a follow-up PR — never auto-resolve "latest"
# because upstream YAML schema is allowed to change between releases and we
# want updates to be deliberate.
_IRON_PROXY_VERSION = "0.39.0"

_IRON_PROXY_RELEASE_BASE = (
    f"https://github.com/ironsh/iron-proxy/releases/download/v{_IRON_PROXY_VERSION}"
)
_IRON_PROXY_CHECKSUM_NAME = "checksums.txt"

# How long to wait for HTTP downloads and subprocess interactions, in seconds.
_DOWNLOAD_TIMEOUT = 120  # binary is ~16MB
_RUN_TIMEOUT = 30
_STARTUP_GRACE_SECONDS = 5

# Default listen ports.  HTTPS_PROXY semantics use a single CONNECT tunnel,
# so we expose only the tunnel listener for v1 — no need to put the sandbox
# DNS at the iron-proxy IP.  This greatly simplifies wiring.
_DEFAULT_TUNNEL_PORT = 9090

# Hosts allowed by default for AI inference traffic.  Anything else is 403'd.
_DEFAULT_ALLOWED_HOSTS: Tuple[str, ...] = (
    "openrouter.ai",
    "*.openrouter.ai",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.x.ai",
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.deepseek.com",
    "inference.nousresearch.com",
)

# Provider env-var name -> upstream host (or list of hosts) on which the
# Authorization Bearer token should be swapped.  Only includes providers
# whose API uses a plain "Authorization: Bearer <key>" header — providers
# with custom auth (x-api-key, query params, signatures) get added as we
# write per-provider rules.
_BEARER_PROVIDERS: Dict[str, Tuple[str, ...]] = {
    "OPENROUTER_API_KEY": ("openrouter.ai", "*.openrouter.ai"),
    "OPENAI_API_KEY": ("api.openai.com",),
    "GROQ_API_KEY": ("api.groq.com",),
    "TOGETHER_API_KEY": ("api.together.xyz",),
    "DEEPSEEK_API_KEY": ("api.deepseek.com",),
    "MISTRAL_API_KEY": ("api.mistral.ai",),
    "XAI_API_KEY": ("api.x.ai",),
    "NOUS_API_KEY": ("inference.nousresearch.com",),
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProxyStatus:
    """Snapshot of the iron-proxy installation + runtime state."""

    enabled: bool = False
    binary_path: Optional[Path] = None
    binary_version: Optional[str] = None
    config_path: Optional[Path] = None
    ca_cert_path: Optional[Path] = None
    pid: Optional[int] = None
    listening: bool = False
    tunnel_port: int = _DEFAULT_TUNNEL_PORT
    warnings: List[str] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        return self.binary_path is not None and self.binary_path.exists()

    @property
    def configured(self) -> bool:
        return (
            self.config_path is not None
            and self.config_path.exists()
            and self.ca_cert_path is not None
            and self.ca_cert_path.exists()
        )


@dataclass
class TokenMapping:
    """Map a sandbox-visible proxy token to a real upstream credential lookup.

    ``real_env_name`` is the env-var name iron-proxy reads at egress time.
    When Bitwarden is configured as the credential source for the proxy,
    iron-proxy's *own* environment is populated from bws on startup — the
    sandbox still sees only ``proxy_token``.
    """

    proxy_token: str
    real_env_name: str
    upstream_hosts: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _hermes_bin_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "bin"


def _proxy_state_dir() -> Path:
    from hermes_constants import get_hermes_home

    d = get_hermes_home() / "proxy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _platform_binary_name() -> str:
    return "iron-proxy.exe" if platform.system() == "Windows" else "iron-proxy"


def _platform_asset_name() -> str:
    """Map (uname, arch) → upstream release asset filename.

    iron-proxy ships ``iron-proxy_<version>_<os>_<arch>.tar.gz``.
    Windows builds aren't published upstream as of v0.39.0; we raise a
    clear error for callers on Windows.
    """

    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        return f"iron-proxy_{_IRON_PROXY_VERSION}_linux_{arch}.tar.gz"
    if system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        return f"iron-proxy_{_IRON_PROXY_VERSION}_darwin_{arch}.tar.gz"
    if system == "Windows":
        raise RuntimeError(
            "iron-proxy does not ship native Windows binaries as of "
            f"v{_IRON_PROXY_VERSION}. Run the proxy on a Linux/macOS host, "
            "or inside WSL."
        )

    raise RuntimeError(
        f"Unsupported platform for iron-proxy auto-install: {system} {machine}"
    )


# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------


def find_iron_proxy(*, install_if_missing: bool = False) -> Optional[Path]:
    """Return a path to a usable ``iron-proxy`` binary, or None.

    Resolution order:
      1. ``<hermes_home>/bin/iron-proxy``  (our managed copy — preferred)
      2. ``shutil.which("iron-proxy")``    (system PATH)

    When ``install_if_missing`` is True and neither resolves, calls
    :func:`install_iron_proxy` to download and verify the pinned version.
    """

    managed = _hermes_bin_dir() / _platform_binary_name()
    if managed.exists() and os.access(managed, os.X_OK):
        return managed

    system = shutil.which("iron-proxy")
    if system:
        return Path(system)

    if install_if_missing:
        try:
            return install_iron_proxy()
        except Exception as exc:  # noqa: BLE001 — never block startup
            logger.warning("iron-proxy auto-install failed: %s", exc)
            return None
    return None


def install_iron_proxy(*, force: bool = False) -> Path:
    """Download, verify, and install the pinned ``iron-proxy`` binary.

    Returns the path to the installed executable.  Raises on any failure
    (network, checksum, extraction).  Callers in the auto-install path catch
    these; the user-facing ``hermes proxy install`` surface lets them
    propagate so the wizard can show a clear error.
    """

    bin_dir = _hermes_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / _platform_binary_name()

    if target.exists() and not force:
        return target

    asset_name = _platform_asset_name()
    asset_url = f"{_IRON_PROXY_RELEASE_BASE}/{asset_name}"
    checksum_url = f"{_IRON_PROXY_RELEASE_BASE}/{_IRON_PROXY_CHECKSUM_NAME}"

    with tempfile.TemporaryDirectory(prefix="hermes-iron-proxy-") as tmpdir:
        tmp = Path(tmpdir)
        archive_path = tmp / asset_name
        checksum_path = tmp / _IRON_PROXY_CHECKSUM_NAME

        logger.info("Downloading %s", asset_url)
        _http_download(asset_url, archive_path)
        _http_download(checksum_url, checksum_path)

        expected = _expected_sha256(checksum_path, asset_name)
        actual = _sha256_file(archive_path)
        if expected.lower() != actual.lower():
            raise RuntimeError(
                f"Checksum mismatch for {asset_name}: "
                f"expected {expected}, got {actual}"
            )

        with tarfile.open(archive_path, "r:gz") as tf:
            member = _pick_tar_member(tf, _platform_binary_name())
            tf.extract(member, tmp)  # noqa: S202 — member name is sanitized below
            extracted = tmp / member.name

        # Stage into the final directory then atomically rename so the new
        # binary is never visible half-written.
        fd, staged = tempfile.mkstemp(dir=str(bin_dir), prefix=".iron-proxy_")
        os.close(fd)
        shutil.copy2(extracted, staged)
        os.chmod(
            staged,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH,
        )
        os.replace(staged, target)

    logger.info("Installed iron-proxy %s at %s", _IRON_PROXY_VERSION, target)
    return target


def _http_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _expected_sha256(checksum_file: Path, asset_name: str) -> str:
    """Parse the standard ``sha256sum`` output: ``<hex>  <filename>``."""

    text = checksum_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == asset_name:
            return parts[0]
    raise RuntimeError(
        f"No checksum entry for {asset_name} in {checksum_file.name}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_tar_member(tf: tarfile.TarFile, binary_name: str) -> tarfile.TarInfo:
    """Find the binary inside the upstream tar.

    iron-proxy's archive is typically flat (binary at root) but we tolerate
    a top-level directory.  Members must be regular files with a leaf name
    matching ``binary_name``, no absolute paths, and no ``..`` traversal.
    """

    candidates: List[tarfile.TarInfo] = []
    for member in tf.getmembers():
        if not member.isfile():
            continue
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            continue
        if Path(member.name).name == binary_name:
            candidates.append(member)
    if not candidates:
        raise RuntimeError(
            f"Could not find {binary_name} inside downloaded archive "
            f"(members: {[m.name for m in tf.getmembers()[:5]]}...)"
        )
    candidates.sort(key=lambda m: len(m.name))
    return candidates[0]


def iron_proxy_version(binary: Path) -> str:
    """Return ``iron-proxy --version`` output, stripped.  Empty on failure."""

    try:
        res = subprocess.run(  # noqa: S603 — binary path is trusted
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (res.stdout or res.stderr or "").strip()


# ---------------------------------------------------------------------------
# CA cert generation
# ---------------------------------------------------------------------------


def ensure_ca_cert(*, force: bool = False) -> Tuple[Path, Path]:
    """Generate (or return existing) iron-proxy CA cert + key.

    Uses the host's ``openssl`` binary.  We don't try to bind to a Python
    crypto library — openssl is universally available on the platforms we
    support, and it sidesteps cryptography-package licensing/distribution
    surface.
    """

    state = _proxy_state_dir()
    ca_crt = state / "ca.crt"
    ca_key = state / "ca.key"

    if ca_crt.exists() and ca_key.exists() and not force:
        return ca_crt, ca_key

    if shutil.which("openssl") is None:
        raise RuntimeError(
            "openssl not found on PATH. Install OpenSSL (apt: `openssl`, "
            "brew: `openssl`) to generate the iron-proxy CA cert."
        )

    # 10-year cert.  iron-proxy mints short-lived leaf certs from this CA,
    # so the CA itself only rotates when the user explicitly forces it.
    with tempfile.TemporaryDirectory(prefix="hermes-proxy-ca-") as tmpdir:
        tmp = Path(tmpdir)
        tmp_key = tmp / "ca.key"
        tmp_crt = tmp / "ca.crt"

        subprocess.run(  # noqa: S603 — openssl path is trusted PATH lookup
            ["openssl", "genrsa", "-out", str(tmp_key), "4096"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(  # noqa: S603
            [
                "openssl", "req", "-x509", "-new", "-nodes",
                "-key", str(tmp_key),
                "-sha256", "-days", "3650",
                "-subj", "/CN=hermes iron-proxy CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign",
                "-out", str(tmp_crt),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

        # Move into place with private permissions.
        shutil.copy2(tmp_key, ca_key)
        shutil.copy2(tmp_crt, ca_crt)
        os.chmod(ca_key, stat.S_IRUSR | stat.S_IWUSR)
        os.chmod(ca_crt, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    logger.info("Generated iron-proxy CA at %s", ca_crt)
    return ca_crt, ca_key


# ---------------------------------------------------------------------------
# Proxy config + token mapping generation
# ---------------------------------------------------------------------------


def mint_proxy_token(prefix: str = "hermes-proxy") -> str:
    """Mint a fresh opaque token to hand to the sandbox.

    The token has no internal structure beyond a recognizable prefix —
    iron-proxy matches on exact equality.  We use a long random suffix
    so collisions are infeasible.
    """

    return f"{prefix}-{hashlib.sha256(os.urandom(32)).hexdigest()[:32]}"


def build_proxy_config(
    *,
    mappings: List[TokenMapping],
    ca_cert: Path,
    ca_key: Path,
    tunnel_port: int = _DEFAULT_TUNNEL_PORT,
    audit_log: Optional[Path] = None,
    allowed_hosts: Optional[List[str]] = None,
    upstream_deny_cidrs: Optional[List[str]] = None,
) -> Dict:
    """Build the iron-proxy YAML config (as a dict) for a given mapping set.

    The dict is YAML-serializable via ``yaml.safe_dump``.  iron-proxy reads
    real secrets from its OWN environment via ``source: {type: env, var: ...}``;
    the sandbox never sees them.

    Schema mirrors the official iron-proxy schema as of v0.39.0.  Notable
    points:

    * The ``dns`` section is required by the binary even when we only use the
      CONNECT tunnel.  We point it at loopback so it doesn't conflict with
      anything else and disable the listener.
    * The ``proxy.tunnel_listen`` is what sandboxes hit via ``HTTPS_PROXY``.
      ``http_listen`` / ``https_listen`` are present (loopback only) so the
      proxy boots; sandboxes never route directly to them.
    * ``allowlist`` transform takes ``domains:`` and ``cidrs:``, not ``hosts:``.
    * ``secrets`` transform takes ``secrets:`` (plural), each with a
      ``source``, a ``replace.proxy_value`` (the sandbox-visible token), and
      a list of ``rules`` saying which hosts the swap should fire on.
    """

    hosts: List[str] = list(allowed_hosts or _DEFAULT_ALLOWED_HOSTS)
    for m in mappings:
        for h in m.upstream_hosts:
            if h not in hosts:
                hosts.append(h)

    secrets_rules = []
    for m in mappings:
        secrets_rules.append({
            "source": {"type": "env", "var": m.real_env_name},
            "replace": {
                "proxy_value": m.proxy_token,
                "match_headers": ["Authorization"],
                # The token is also accepted as a bearer query param in case
                # the sandbox passes it that way.  Body matching is off — we
                # don't want body inspection forced for every request.
                "match_query": True,
                "match_body": False,
            },
            "rules": [{"host": h} for h in m.upstream_hosts],
        })

    return {
        # DNS section is required by the binary's config parser, but we run
        # in tunnel-only mode so the DNS listener never binds an exposed port.
        # Sandboxes reach the proxy via HTTPS_PROXY/CONNECT, not via DNS
        # redirection.
        "dns": {
            "listen": "127.0.0.1:0",   # ephemeral loopback — effectively disabled
            "proxy_ip": "127.0.0.1",
        },
        "proxy": {
            # http_listen is the HTTP-proxy listener that handles both plain
            # HTTP forwards AND CONNECT tunnels for HTTPS.  Sandboxes set
            # `HTTPS_PROXY=http://host:tunnel_port` and the same listener
            # serves both protocols.  Bind on all interfaces so containers
            # can reach it via host.docker.internal.
            "http_listen": f":{tunnel_port}",
            # The HTTPS-listener (direct TLS termination, no CONNECT) and
            # the SOCKS5/CONNECT-only tunnel listener get loopback ephemeral
            # ports — we don't expose them.
            "https_listen": "127.0.0.1:0",
            "tunnel_listen": "127.0.0.1:0",
            "max_request_body_bytes": 16 * 1024 * 1024,
            "max_response_body_bytes": 0,
            "upstream_response_header_timeout": "120s",
            # SSRF protection: deny outbound to cloud metadata + loopback by
            # default.  Tests / dev setups that need loopback can pass an
            # explicit override (e.g. [] to disable, or just the IMDS subset).
            **(
                {"upstream_deny_cidrs": list(upstream_deny_cidrs)}
                if upstream_deny_cidrs is not None
                else {}
            ),
        },
        "tls": {
            "ca_cert": str(ca_cert),
            "ca_key": str(ca_key),
            "cert_cache_size": 1000,
            "leaf_cert_expiry_hours": 168,
        },
        "transforms": [
            {
                "name": "allowlist",
                "config": {"domains": hosts},
            },
            {
                "name": "secrets",
                "config": {"secrets": secrets_rules},
            },
        ],
        "log": {"level": "info"},
    }


def write_proxy_config(config: Dict) -> Path:
    """Serialize the config dict to ``<hermes_home>/proxy/proxy.yaml``.

    Uses ``yaml.safe_dump`` so we never emit Python tags.
    """

    try:
        import yaml  # PyYAML is already a Hermes dep
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to write the iron-proxy config but is not "
            "installed."
        ) from exc

    state = _proxy_state_dir()
    out = state / "proxy.yaml"
    tmp_path = state / ".proxy.yaml.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp_path, out)
    os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
    return out


def write_mappings(mappings: List[TokenMapping]) -> Path:
    """Persist the sandbox-visible proxy tokens to ``mappings.json``.

    The Docker backend reads this file to inject the right tokens as env
    vars when starting a sandbox.  The file is NOT read by iron-proxy
    itself — the mapping is already baked into ``proxy.yaml``.
    """

    state = _proxy_state_dir()
    out = state / "mappings.json"
    payload = {
        "version": 1,
        "tokens": [
            {
                "proxy_token": m.proxy_token,
                "env_name": m.real_env_name,
                "upstream_hosts": list(m.upstream_hosts),
            }
            for m in mappings
        ],
    }
    tmp_path = state / ".mappings.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out)
    os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
    return out


def load_mappings() -> List[TokenMapping]:
    """Read mappings.json, if it exists.  Empty list on any error."""

    state = _proxy_state_dir()
    f = state / "mappings.json"
    if not f.exists():
        return []
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read iron-proxy mappings.json: %s", exc)
        return []
    out: List[TokenMapping] = []
    for item in payload.get("tokens", []):
        try:
            out.append(TokenMapping(
                proxy_token=item["proxy_token"],
                real_env_name=item["env_name"],
                upstream_hosts=tuple(item.get("upstream_hosts") or ()),
            ))
        except (KeyError, TypeError):
            continue
    return out


def discover_provider_mappings(
    *,
    available_env_names: Optional[List[str]] = None,
) -> List[TokenMapping]:
    """Mint a TokenMapping for every known provider whose env var is set.

    Pass ``available_env_names`` to override the lookup source (used by the
    Bitwarden adapter so we mint mappings for keys that *will* be in the
    proxy's environment even if they aren't in the host process env right
    now).
    """

    if available_env_names is not None:
        names = set(available_env_names)
    else:
        names = {k for k, v in os.environ.items() if v}

    mappings: List[TokenMapping] = []
    for env_name, hosts in _BEARER_PROVIDERS.items():
        if env_name not in names:
            continue
        mappings.append(TokenMapping(
            proxy_token=mint_proxy_token(prefix=env_name.lower().replace("_api_key", "")),
            real_env_name=env_name,
            upstream_hosts=hosts,
        ))
    return mappings


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def _pidfile() -> Path:
    return _proxy_state_dir() / "iron-proxy.pid"


def _read_pid() -> Optional[int]:
    pf = _pidfile()
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` is alive AND is an iron-proxy process.

    The cmdline check guards against PID reuse — without it, an unrelated
    process that happens to have grabbed the same PID after iron-proxy
    crashed would look "alive" and we'd refuse to restart.
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False

    # Confirm via /proc when available (Linux + WSL).  On macOS we fall
    # back to ``ps``; on truly exotic platforms the os.kill(pid, 0) result
    # is taken at face value.
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            cmdline = cmdline_path.read_bytes().decode("utf-8", errors="ignore")
            return "iron-proxy" in cmdline
    except OSError:
        pass
    try:
        res = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=2,
        )
        if res.returncode == 0:
            return "iron-proxy" in (res.stdout or "")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return True


def start_proxy(
    *,
    binary: Optional[Path] = None,
    config_path: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> ProxyStatus:
    """Spawn iron-proxy as a managed background subprocess.

    Idempotent — if the proxy is already running with the expected PID,
    just returns the live status.
    """

    existing = _read_pid()
    if existing and _pid_alive(existing):
        return get_status()

    bin_path = binary or find_iron_proxy(install_if_missing=True)
    if bin_path is None:
        raise RuntimeError(
            "iron-proxy binary not available — run `hermes proxy install`."
        )

    cfg = config_path or (_proxy_state_dir() / "proxy.yaml")
    if not cfg.exists():
        raise RuntimeError(
            f"iron-proxy config not found at {cfg}. "
            "Run `hermes proxy setup` first."
        )

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("NO_COLOR", "1")

    log_path = _proxy_state_dir() / "iron-proxy.log"
    log_fp = open(log_path, "ab", buffering=0)

    try:
        proc = subprocess.Popen(  # noqa: S603 — binary path is trusted
            [str(bin_path), "-config", str(cfg)],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_fp.close()
        raise RuntimeError(f"failed to spawn iron-proxy: {exc}") from exc

    # Give it a moment to either come up or fail fast.  We don't want to
    # write a pidfile pointing at a dead process.
    time.sleep(_STARTUP_GRACE_SECONDS)
    if proc.poll() is not None:
        log_fp.close()
        tail = _tail_log(log_path, lines=20)
        raise RuntimeError(
            f"iron-proxy exited immediately (code {proc.returncode}). "
            f"Last log lines:\n{tail}"
        )

    pidfile = _pidfile()
    pidfile.write_text(str(proc.pid), encoding="utf-8")
    logger.info("Started iron-proxy pid=%s config=%s", proc.pid, cfg)
    return get_status()


def stop_proxy() -> bool:
    """Stop the managed iron-proxy.  Returns True if it was running."""

    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        _pidfile().unlink(missing_ok=True)
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _pidfile().unlink(missing_ok=True)
        return False

    # Wait up to 5s for graceful exit, then SIGKILL.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _pidfile().unlink(missing_ok=True)
    logger.info("Stopped iron-proxy pid=%s", pid)
    return True


def get_status() -> ProxyStatus:
    """Snapshot the current proxy state — does NOT start anything."""

    status = ProxyStatus()
    status.tunnel_port = _read_tunnel_port_from_config() or _DEFAULT_TUNNEL_PORT

    binary = find_iron_proxy(install_if_missing=False)
    if binary:
        status.binary_path = binary
        status.binary_version = iron_proxy_version(binary)

    state = _proxy_state_dir()
    cfg = state / "proxy.yaml"
    ca = state / "ca.crt"
    if cfg.exists():
        status.config_path = cfg
    if ca.exists():
        status.ca_cert_path = ca

    pid = _read_pid()
    if pid and _pid_alive(pid):
        status.pid = pid
        status.listening = _port_listening("127.0.0.1", status.tunnel_port)

    return status


def _read_tunnel_port_from_config() -> Optional[int]:
    cfg = _proxy_state_dir() / "proxy.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    # The CLI/Docker side calls this "the tunnel port" because that's how
    # sandboxes use it (HTTPS_PROXY), but on the iron-proxy side it's the
    # http_listen — the HTTP-proxy listener handles both plain HTTP and the
    # CONNECT method for HTTPS upstreams.
    listen = ((data or {}).get("proxy") or {}).get("http_listen") or ""
    if not isinstance(listen, str) or ":" not in listen:
        return None
    try:
        return int(listen.rsplit(":", 1)[1])
    except ValueError:
        return None


def _port_listening(host: str, port: int) -> bool:
    """Cheap TCP connect probe — True iff something accepts on host:port."""

    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _tail_log(path: Path, *, lines: int = 20) -> str:
    if not path.exists():
        return "(no log file)"
    try:
        data = path.read_bytes()[-8192:]
        return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"(could not read log: {exc})"


# ---------------------------------------------------------------------------
# Test hook
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """No-op today — kept symmetric with bitwarden._reset_cache_for_tests."""

    return None


# Make a small set of symbols available without underscored access.
__all__ = [
    "ProxyStatus",
    "TokenMapping",
    "build_proxy_config",
    "discover_provider_mappings",
    "ensure_ca_cert",
    "find_iron_proxy",
    "get_status",
    "install_iron_proxy",
    "iron_proxy_version",
    "load_mappings",
    "mint_proxy_token",
    "start_proxy",
    "stop_proxy",
    "write_mappings",
    "write_proxy_config",
]
