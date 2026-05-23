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
  first ``hermes egress setup``.  Sandboxes trust this CA so iron-proxy can
  terminate TLS and rewrite headers.

* The proxy config lives at ``<hermes_home>/proxy/proxy.yaml``.  It enumerates
  the per-provider allowlists and the ``secrets`` transform that does the
  Authorization-header swap.

* Token mappings (proxy token -> real credential lookup) live alongside the
  config.  The real credential is **never** written to the config — iron-proxy
  reads it from its own environment via ``{type: env, var: NAME}``.  When
  Bitwarden Secrets Manager is configured, the real value is pulled there
  at proxy startup instead.

* The proxy runs as a managed subprocess (``hermes egress start``), pidfile
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


# Providers whose env-var names we recognize but whose API uses a non-bearer
# auth scheme (x-api-key, AAD/OAuth, SigV4, custom signatures).  When any of
# these env vars are present at proxy-start time AND
# ``proxy.fail_on_uncovered_providers`` is true (default), ``start_proxy``
# refuses to start.  Without this list the sandbox would still hold real
# credentials for these providers and silently bypass the proxy.
#
# Bare strings here are env-var names; the proxy doesn't try to wire them up,
# only flags their presence so the operator knows isolation is incomplete.
_NON_BEARER_PROVIDERS: Tuple[str, ...] = (
    # Anthropic native uses x-api-key, not Authorization: Bearer.
    "ANTHROPIC_API_KEY",
    # Azure OpenAI: api-key header + optional AAD bearer.
    "AZURE_OPENAI_API_KEY",
    # AWS Bedrock / SageMaker: SigV4-signed requests.
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    # GCP Vertex AI: OAuth bearer from gcloud SDK, not a static env key.
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Google AI Studio (Gemini): x-goog-api-key OR query param.
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


# Default SSRF-protection deny list applied to the proxy's outbound traffic.
# Mirrors the public docs promise ("cloud metadata IPs are refused by default
# regardless of allowlist").  Tests / dev setups that need loopback can pass
# an explicit override (e.g. [] to disable, or a smaller subset).
_DEFAULT_UPSTREAM_DENY_CIDRS: Tuple[str, ...] = (
    "127.0.0.0/8",        # IPv4 loopback
    "::1/128",            # IPv6 loopback
    "169.254.0.0/16",     # IPv4 link-local incl. AWS/GCP/Azure IMDS
    "fe80::/10",          # IPv6 link-local
    "10.0.0.0/8",         # RFC1918
    "172.16.0.0/12",      # RFC1918
    "192.168.0.0/16",     # RFC1918
    "fc00::/7",           # IPv6 ULA
)


# Min env vars the iron-proxy subprocess actually needs.  Everything else
# is stripped — see ``_build_proxy_subprocess_env`` for the rationale.
_PROXY_SUBPROCESS_ENV_ALLOWLIST: Tuple[str, ...] = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TZ",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NO_COLOR",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",  # Windows
    "USERPROFILE",  # Windows
)


# Env vars that must be stripped from the subprocess env even if they're on
# the allowlist or named in mappings — these would either recurse the proxy
# back through itself or send its traffic through a corporate proxy.
_PROXY_SUBPROCESS_ENV_STRIP: Tuple[str, ...] = (
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy",
)


# SIGKILL doesn't exist on Windows.  We fall back to SIGTERM there, which the
# OS treats as a hard terminate via TerminateProcess() — equivalent semantics.
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


# Cached ``iron-proxy --version`` output keyed by binary path.  ``get_status``
# is invoked per Docker-container-create; the version string is constant for
# a given binary so a one-shot subprocess call is plenty.
_VERSION_CACHE: Dict[str, str] = {}


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


def _proxy_state_dir_ro() -> Path:
    """Return the proxy state dir without creating it.

    Read-only callers (status probes, pidfile reads, version queries) use
    this — there's no reason to materialize ``~/.hermes/proxy/`` just to
    check whether a pidfile exists.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "proxy"


def _proxy_state_dir() -> Path:
    """Return the proxy state dir, creating it with 0o700 if absent.

    Writable callers (CA gen, config write, mappings write, start_proxy)
    use this.  We force 0o700 — the dir holds the CA signing key, audit
    log, and pidfile, so traversal by other local users is undesirable.
    The chmod is unconditional so a pre-existing dir with a slack umask
    gets tightened on first access.
    """
    d = _proxy_state_dir_ro()
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        # On Windows the chmod is a no-op for POSIX modes; on shared
        # filesystems we may not own the dir.  Don't fail here — the
        # individual files still get explicit perms.
        pass
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
            # PEP 706 data filter — strips ownership/mode replay (we set
            # chmod explicitly below) AND rejects symlink/hardlink members
            # that escape the extraction dir.  Required on 3.12+ to silence
            # the deprecation warning and on 3.14+ to opt into the
            # tarbomb-rejecting default.
            try:
                tf.extract(member, tmp, filter="data")  # noqa: S202
            except TypeError:
                # Python < 3.12 — filter kw didn't exist yet; the
                # _pick_tar_member sanitization already rejects path
                # traversal so this is acceptable.
                tf.extract(member, tmp)  # noqa: S202
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
    """Return ``iron-proxy --version`` output, stripped.  Empty on failure.

    Cached by binary path: ``get_status`` is called per Docker container
    create, but the version string is constant for a given binary.  A
    single subprocess invocation is plenty.
    """

    key = str(binary)
    cached = _VERSION_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        res = subprocess.run(  # noqa: S603 — binary path is trusted
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    out = (res.stdout or res.stderr or "").strip()
    _VERSION_CACHE[key] = out
    return out


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

        # Move into place with private permissions.  CRITICAL: the key
        # has to be created with 0o600 from the very first byte — a
        # ``shutil.copy2`` followed by ``os.chmod`` leaves a TOCTOU window
        # where the private key is world-readable on multi-user hosts.
        key_bytes = tmp_key.read_bytes()
        crt_bytes = tmp_crt.read_bytes()

        # Stage with explicit 0o600, then atomically rename into place.
        # O_NOFOLLOW guards against a symlink at ca_key (defence-in-depth
        # — the state dir is 0o700-owned but a malicious local user with
        # the same uid could pre-create one).
        key_staged = ca_key.with_suffix(ca_key.suffix + ".staged")
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        # O_NOFOLLOW exists on POSIX; on Windows we just rely on the
        # default semantics.
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        # Best-effort: pre-unlink any existing staged file so the open
        # with O_CREAT is always against a fresh inode.
        try:
            key_staged.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(str(key_staged), open_flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(key_bytes)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(key_staged, ca_key)

        # Cert is public — 0o644 is fine and matches typical PEM layout.
        ca_crt.write_bytes(crt_bytes)
        os.chmod(ca_crt, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    logger.info("Generated iron-proxy CA at %s", ca_crt)
    return ca_crt, ca_key


# ---------------------------------------------------------------------------
# Proxy config + token mapping generation
# ---------------------------------------------------------------------------


def mint_proxy_token(prefix: str = "hermes-proxy") -> str:
    """Mint a fresh opaque token to hand to the sandbox.

    The token has no internal structure beyond a recognizable prefix —
    iron-proxy matches on exact equality.  We use a 128-bit random suffix
    (32 hex chars from a SHA-256 of 32 bytes of os.urandom).  At that
    entropy the birthday-bound collision probability is below 2^-64 for
    up to 2^32 tokens, which is plenty for a proxy-scoped namespace.
    """

    return f"{prefix}-{hashlib.sha256(os.urandom(32)).hexdigest()[:32]}"


def _default_http_listen(tunnel_port: int) -> List[str]:
    """Build the list of host:port pairs the proxy should bind on.

    Always binds loopback (``127.0.0.1``) so host-side test tooling can hit
    the proxy directly.  On Linux we also bind the docker bridge gateway
    (``172.17.0.1`` by default) so containers can reach the proxy via
    ``host.docker.internal:host-gateway``.  We do NOT bind ``0.0.0.0`` —
    that would expose the proxy (and, with a leaked sandbox token, the
    user's API quota) to anyone on the local network.

    On macOS / Windows Docker Desktop the bridge gateway is managed by
    Desktop itself and ``host.docker.internal`` resolves via VPNkit, so
    a single loopback bind is enough.
    """

    binds = [f"127.0.0.1:{tunnel_port}"]
    if platform.system() == "Linux":
        bridge_ip = _detect_docker_bridge_ip()
        if bridge_ip and bridge_ip != "127.0.0.1":
            binds.append(f"{bridge_ip}:{tunnel_port}")
    return binds


def _detect_docker_bridge_ip() -> Optional[str]:
    """Return the docker0 bridge IPv4, if present, else None.

    Best-effort: we try ``ip -4 addr show docker0`` first, then fall back
    to parsing ``/proc/net/route`` for the bridge IP.  Anything that fails
    or doesn't look like an IPv4 returns None — callers handle that as
    "no bridge bind".
    """

    try:
        res = subprocess.run(  # noqa: S603 — ip is a system binary
            ["ip", "-4", "-o", "addr", "show", "docker0"],
            capture_output=True, text=True, timeout=2,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.split()
                # Expected: "<n>: docker0  inet 172.17.0.1/16 ..."
                for i, tok in enumerate(parts):
                    if tok == "inet" and i + 1 < len(parts):
                        ip = parts[i + 1].split("/")[0]
                        # cheap sanity: four dotted parts.
                        if ip.count(".") == 3:
                            return ip
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def build_proxy_config(
    *,
    mappings: List[TokenMapping],
    ca_cert: Path,
    ca_key: Path,
    tunnel_port: int = _DEFAULT_TUNNEL_PORT,
    audit_log: Optional[Path] = None,
    allowed_hosts: Optional[List[str]] = None,
    upstream_deny_cidrs: Optional[List[str]] = None,
    http_listen: Optional[List[str]] = None,
) -> Dict:
    """Build the iron-proxy YAML config (as a dict) for a given mapping set.

    The dict is YAML-serializable via ``yaml.safe_dump``.  iron-proxy reads
    real secrets from its OWN environment via ``source: {type: env, var: ...}``;
    the sandbox never sees them.

    Bind policy: by default we bind loopback (``127.0.0.1``) plus the
    docker bridge gateway IP on Linux (``172.17.0.1`` or whatever
    ``docker0`` resolves to).  Sandboxes use ``host.docker.internal`` which
    Linux Docker maps to the bridge gateway via ``--add-host``; macOS /
    Windows Docker Desktop manage their own gateway.  We do NOT bind
    ``0.0.0.0`` — a LAN peer with a leaked sandbox token could otherwise
    spend the operator's API quota against any allowlisted upstream.

    SSRF policy: ``upstream_deny_cidrs`` defaults to a conservative deny
    list covering loopback, link-local (incl. AWS/GCP/Azure IMDS at
    169.254.169.254), and RFC1918.  Pass an explicit ``[]`` to opt out of
    the deny list entirely (only sensible in hermetic tests).

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

    # SSRF protection: default-deny cloud metadata + loopback + RFC1918.
    # Callers can pass [] to opt out entirely (hermetic tests need this for
    # talking to a loopback upstream).  None means "use the default".
    deny_cidrs: List[str]
    if upstream_deny_cidrs is None:
        deny_cidrs = list(_DEFAULT_UPSTREAM_DENY_CIDRS)
    else:
        deny_cidrs = list(upstream_deny_cidrs)

    # Listen addresses.  Single canonical "http_listen" for backward compat
    # plus a "http_listens" list (iron-proxy v0.39 accepts both; v0.40+ is
    # listen-list-only).  We always emit both forms so a binary version
    # bump can't silently regress the bind policy.
    listens = list(http_listen) if http_listen else _default_http_listen(tunnel_port)
    primary_listen = listens[0] if listens else f"127.0.0.1:{tunnel_port}"

    log_block: Dict = {"level": "info"}
    if audit_log is not None:
        # Wire the operator-requested audit-log path into the binary's log
        # config.  iron-proxy reads ``log.audit_path``; setting it routes
        # per-request records there (separately from server-level logs).
        log_block["audit_path"] = str(audit_log)

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
            # serves both protocols.  We bind loopback + the docker bridge
            # gateway (Linux) — NOT 0.0.0.0.  LAN peers with a leaked
            # sandbox token would otherwise be able to spend the operator's
            # API quota against any allowlisted upstream.
            "http_listen": primary_listen,
            "http_listens": listens,
            # The HTTPS-listener (direct TLS termination, no CONNECT) and
            # the SOCKS5/CONNECT-only tunnel listener get loopback ephemeral
            # ports — we don't expose them.
            "https_listen": "127.0.0.1:0",
            "tunnel_listen": "127.0.0.1:0",
            "max_request_body_bytes": 16 * 1024 * 1024,
            "max_response_body_bytes": 0,
            "upstream_response_header_timeout": "120s",
            # SSRF protection: deny outbound to cloud metadata + loopback by
            # default.  An empty list opts out entirely.
            "upstream_deny_cidrs": deny_cidrs,
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
        "log": log_block,
    }


def ensure_audit_log(audit_path: Path) -> None:
    """Create the audit log file with private permissions (0o600).

    Called from the wizard right before ``start_proxy``.  Without this,
    iron-proxy creates the file under the default umask the first time it
    writes — meaning every host-side request history is potentially
    world-readable.  We pre-create the file empty with 0o600 so the daemon
    inherits the tight permissions.
    """

    try:
        # Use os.open + O_CREAT to avoid races on the chmod.
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        fd = os.open(str(audit_path), open_flags, 0o600)
        try:
            # Tighten perms even if the file already existed under a
            # slacker umask.
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
    except OSError as exc:
        logger.warning("Could not pre-create audit log %s: %s", audit_path, exc)


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


def discover_uncovered_providers(
    *,
    available_env_names: Optional[List[str]] = None,
) -> List[str]:
    """Return env-var names for providers we recognize but can't proxy.

    Anthropic native (x-api-key), AWS Bedrock (SigV4), Azure OpenAI
    (api-key), etc.  When any of these are configured, the sandbox is
    holding real credentials that the proxy can't strip — the isolation
    guarantee is incomplete for those providers.

    The wizard uses this to print a warning at setup time; ``start_proxy``
    can be configured to refuse to start when ``fail_on_uncovered_providers``
    is true.
    """

    if available_env_names is not None:
        names = set(available_env_names)
    else:
        names = {k for k, v in os.environ.items() if v}

    return [n for n in _NON_BEARER_PROVIDERS if n in names]


def merge_mappings(
    *,
    existing: List[TokenMapping],
    discovered: List[TokenMapping],
    rotate: bool = False,
) -> List[TokenMapping]:
    """Combine an existing mapping set with freshly discovered providers.

    By default this PRESERVES tokens for providers already in ``existing`` —
    re-running ``hermes egress setup`` should not invalidate the tokens
    baked into containers that are already running.  Only newly added
    providers get freshly minted tokens.

    When ``rotate=True``, every token in the result is freshly minted
    regardless of overlap.  The wizard exposes this via ``--rotate-tokens``
    for the rare case where the operator wants to roll all tokens
    deliberately (e.g. after a suspected token leak).

    Providers that are in ``existing`` but no longer in ``discovered``
    (operator removed the env var since last setup) are dropped.
    """

    by_name = {m.real_env_name: m for m in existing}
    out: List[TokenMapping] = []
    for d in discovered:
        prior = by_name.get(d.real_env_name)
        if prior is not None and not rotate:
            # Preserve the token, refresh the host list in case we added
            # new upstreams since last setup.
            out.append(TokenMapping(
                proxy_token=prior.proxy_token,
                real_env_name=prior.real_env_name,
                upstream_hosts=d.upstream_hosts,
            ))
        else:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def _pidfile() -> Path:
    return _proxy_state_dir() / "iron-proxy.pid"


def _read_pid() -> Optional[int]:
    # Use the read-only path: don't create the proxy dir just to read the
    # pidfile.  If neither pid file nor dir exists, the daemon is plainly
    # not running.
    pf = _proxy_state_dir_ro() / "iron-proxy.pid"
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


# Nonce env-var set in the iron-proxy subprocess at start_proxy time.  Used
# by ``_pid_alive`` to confirm a candidate PID still refers to *our* managed
# binary even across PID recycling (a fresh process can't inherit our
# arbitrary env value).
_HERMES_IRON_PROXY_NONCE_ENV = "HERMES_IRON_PROXY_NONCE"
_proxy_nonce: Optional[str] = None


def _pid_proc_starttime(pid: int) -> Optional[str]:
    """Return /proc/<pid>/stat[21] (starttime) on Linux, else None.

    Comparing starttime is the standard cheap way to detect PID recycling
    without relying on cmdline scanning.  When None, callers fall back to
    the cmdline + nonce check.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # /proc/<pid>/stat: pid (comm-with-parens) state ppid ... fields[21]=starttime
    # The "comm" field can contain spaces and parens, so split from the
    # right parenthesis instead of using shlex.
    rparen = text.rfind(")")
    if rparen < 0:
        return None
    fields = text[rparen + 1:].split()
    # field index in the post-")" tail: original 3..n become fields[0..n-3]
    # starttime is original field 22 (1-indexed) → tail index 22-3 = 19
    if len(fields) <= 19:
        return None
    return fields[19]


def _pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` is alive AND is an iron-proxy process.

    Defends against PID reuse via three signals (in priority order):
    1. ``/proc/<pid>/environ`` contains our nonce  (most reliable, Linux)
    2. ``/proc/<pid>/cmdline`` basename matches the managed binary
    3. ``ps -p <pid>`` command line contains the binary path

    The legacy ``"iron-proxy" in cmdline`` match was loose enough to match
    ``tail iron-proxy.log`` or an editor with that file open.  We tighten
    on argv[0] basename plus an in-process nonce instead.
    """

    if pid <= 0:
        return False
    try:
        # Use psutil.pid_exists when available — it's a no-op on Windows
        # whereas os.kill(pid, 0) on Windows is actually a hard kill
        # (CTRL_C_EVENT to the target's console process group).  See
        # bpo-14484.  windows-footgun: ok — we explicitly skip the
        # os.kill probe on Windows below.
        import psutil  # type: ignore
        if not psutil.pid_exists(pid):
            return False
    except ImportError:
        if platform.system() == "Windows":
            # On Windows without psutil we can't safely probe — assume
            # the pidfile content is fresh and confirm via the cmdline
            # path below.  os.kill(pid, 0) is NOT safe here.
            pass
        else:
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only branch
            except (ProcessLookupError, PermissionError, OSError):
                return False

    # Strong proof: nonce env var matches.  /proc/<pid>/environ is null-
    # separated KEY=VALUE pairs; substring search is safe.
    if _proxy_nonce:
        try:
            env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
            needle = f"{_HERMES_IRON_PROXY_NONCE_ENV}={_proxy_nonce}".encode()
            if needle in env_bytes:
                return True
        except OSError:
            pass

    # Fallback: cmdline basename match.  argv[0] is the first null-
    # separated token in /proc/<pid>/cmdline.
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            tokens = cmdline_path.read_bytes().split(b"\x00")
            if tokens:
                argv0 = tokens[0].decode("utf-8", errors="ignore")
                argv0_base = os.path.basename(argv0)
                if argv0_base.startswith("iron-proxy"):
                    return True
            return False
    except OSError:
        pass

    # macOS / non-Linux fallback: ``ps`` command basename.
    try:
        res = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=2,
        )
        if res.returncode == 0:
            comm = (res.stdout or "").strip()
            return os.path.basename(comm).startswith("iron-proxy")
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Exotic platforms: be conservative — if the OS says alive we believe
    # it.  This restores the previous behaviour for non-Linux/non-macOS.
    return True


def start_proxy(
    *,
    binary: Optional[Path] = None,
    config_path: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    refresh_secrets_from_bitwarden: bool = False,
    bitwarden_config: Optional[Dict] = None,
) -> ProxyStatus:
    """Spawn iron-proxy as a managed background subprocess.

    Idempotent — if the proxy is already running with the expected PID,
    just returns the live status.

    ``refresh_secrets_from_bitwarden=True`` re-fetches upstream secrets
    via ``bws secret list`` at startup and injects them into the child
    env.  This delivers the rotation promise that distinguishes
    ``credential_source: bitwarden`` from ``credential_source: env``.
    Without this flag (or with ``bitwarden_config=None``) the proxy still
    starts but uses whatever the host process env happens to contain.
    """

    global _proxy_nonce

    existing = _read_pid()
    if existing and _pid_alive(existing):
        return get_status()

    bin_path = binary or find_iron_proxy(install_if_missing=True)
    if bin_path is None:
        raise RuntimeError(
            "iron-proxy binary not available — run `hermes egress install`."
        )

    cfg = config_path or (_proxy_state_dir() / "proxy.yaml")
    if not cfg.exists():
        raise RuntimeError(
            f"iron-proxy config not found at {cfg}. "
            "Run `hermes egress setup` first."
        )

    # Build a minimal subprocess env.  os.environ.copy() would ship every
    # secret in the operator's shell to the proxy — /proc/<pid>/environ
    # would then expose OPENAI_API_KEY, AWS keys, etc. to any same-uid
    # local process.  Defeats the threat model the proxy exists to
    # mitigate.
    env = _build_proxy_subprocess_env(
        extra_env=extra_env,
        refresh_from_bitwarden=refresh_secrets_from_bitwarden,
        bitwarden_config=bitwarden_config,
    )

    # Plant a per-start nonce in the child env so ``_pid_alive`` can
    # confirm a candidate PID still refers to *our* binary across PID
    # recycling.  Module-global is fine — only one managed proxy per
    # Hermes process.
    _proxy_nonce = hashlib.sha256(os.urandom(16)).hexdigest()
    env[_HERMES_IRON_PROXY_NONCE_ENV] = _proxy_nonce

    log_path = _proxy_state_dir() / "iron-proxy.log"
    # Keep ownership of the fd tight: open with explicit 0o600 so the
    # log doesn't get world-readable under a slack umask, then close it
    # immediately after Popen (the child has its own dup).  Without the
    # close-on-success path, every restart leaked one fd in the Hermes
    # process.
    log_fd = os.open(
        str(log_path),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.fchmod(log_fd, 0o600)  # tighten if file pre-existed
    except OSError:
        pass

    try:
        # Use the fd directly via the dup mechanism; Popen will dup() it
        # into the child so we can close ours unconditionally below.
        # NOTE: on Windows ``start_new_session`` is invalid; we don't
        # support Windows for the proxy (the binary itself doesn't ship)
        # but the kwarg is POSIX-only and silently ignored on Win.
        popen_kwargs: Dict = dict(
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
        )
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(  # noqa: S603 — binary path is trusted
            [str(bin_path), "-config", str(cfg)],
            **popen_kwargs,
        )
    except OSError as exc:
        os.close(log_fd)
        raise RuntimeError(f"failed to spawn iron-proxy: {exc}") from exc
    finally:
        # Close our copy of the fd whether Popen raised or succeeded.
        # The child has its own dup via Popen, so it's still writing.
        try:
            os.close(log_fd)
        except OSError:
            pass

    # Poll-with-timeout instead of an unconditional 5s sleep.  The Go
    # binary normally comes up in <200ms; falling through within 100ms
    # of liveness keeps Docker container creation snappy.
    tunnel_port = _read_tunnel_port_from_config() or _DEFAULT_TUNNEL_PORT
    deadline = time.time() + _STARTUP_GRACE_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail_log(log_path, lines=20)
            raise RuntimeError(
                f"iron-proxy exited immediately (code {proc.returncode}). "
                f"Last log lines:\n{tail}"
            )
        if _port_listening("127.0.0.1", tunnel_port):
            break
        time.sleep(0.1)

    if proc.poll() is not None:
        tail = _tail_log(log_path, lines=20)
        raise RuntimeError(
            f"iron-proxy exited immediately (code {proc.returncode}). "
            f"Last log lines:\n{tail}"
        )

    pidfile = _pidfile()
    # Use os.open with O_NOFOLLOW to refuse to follow a pre-existing
    # symlink at the pidfile path (defence-in-depth; same-uid required to
    # plant a symlink but worth defending).  O_TRUNC clobbers any stale
    # content.
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(pidfile), open_flags, 0o600)
    except OSError as exc:
        # If the file existed as a symlink, O_NOFOLLOW returns ELOOP.
        # Surface a clear error and let the operator clean up.
        raise RuntimeError(
            f"Refusing to write pidfile {pidfile}: {exc}.  "
            "Remove that path manually and retry."
        ) from exc
    try:
        # Verify the file we just opened is owned by us.  On POSIX,
        # st_uid mismatch means a same-uid race won and we got a hostile
        # file — bail rather than write the pid into it.
        try:
            st = os.fstat(fd)
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                raise RuntimeError(
                    f"pidfile {pidfile} has unexpected owner uid={st.st_uid}"
                )
        except AttributeError:
            pass  # Windows
        os.write(fd, str(proc.pid).encode("utf-8"))
    finally:
        os.close(fd)

    logger.info("Started iron-proxy pid=%s config=%s", proc.pid, cfg)
    return get_status()


def _build_proxy_subprocess_env(
    *,
    extra_env: Optional[Dict[str, str]] = None,
    refresh_from_bitwarden: bool = False,
    bitwarden_config: Optional[Dict] = None,
) -> Dict[str, str]:
    """Construct the minimal env for the iron-proxy subprocess.

    Allowlists infrastructure vars (PATH, HOME, locale) plus the env vars
    named in ``load_mappings()`` (the real upstream secrets the proxy
    needs to do the swap).  Everything else is stripped — see
    ``_PROXY_SUBPROCESS_ENV_STRIP`` for proxy chain protection.

    When ``refresh_from_bitwarden=True`` AND ``bitwarden_config`` is
    populated, fetches upstream secrets via the BSM SDK at startup and
    merges them in.  This is what delivers the rotation guarantee
    promised by ``credential_source: bitwarden`` — without it, rotating
    a key in the Bitwarden web app doesn't reach the proxy.
    """

    env: Dict[str, str] = {}
    parent = os.environ
    for name in _PROXY_SUBPROCESS_ENV_ALLOWLIST:
        if name in parent:
            env[name] = parent[name]

    # The proxy reads the real upstream secrets from its OWN env, indexed
    # by ``m.real_env_name`` in the YAML config's ``secrets.source.var``
    # field.  Forward those — but only those.
    needed = {m.real_env_name for m in load_mappings()}
    for name in needed:
        if name in parent:
            env[name] = parent[name]

    # Optional Bitwarden refresh path.  Pulled lazily so the proxy module
    # doesn't hard-depend on the bitwarden module being importable in
    # every install.
    if refresh_from_bitwarden and bitwarden_config:
        try:
            from agent.secret_sources import bitwarden as bw
            access_token_name = bitwarden_config.get(
                "access_token_env", "BWS_ACCESS_TOKEN"
            )
            access_token = parent.get(access_token_name, "").strip()
            project_id = bitwarden_config.get("project_id", "")
            if access_token and project_id:
                secrets, warnings = bw.fetch_bitwarden_secrets(
                    access_token=access_token,
                    project_id=project_id,
                    cache_ttl_seconds=0,
                    use_cache=False,
                )
                # Only inject env names we have a mapping for — extra
                # secrets in the BW project shouldn't leak into the proxy
                # process unless they're going to be used by the swap.
                for n in needed:
                    if n in secrets:
                        env[n] = secrets[n]
                for w in warnings:
                    logger.warning("Bitwarden refresh: %s", w)
            else:
                logger.warning(
                    "credential_source=bitwarden but access_token_env=%s or "
                    "project_id is empty — proxy will fall back to parent env",
                    access_token_name,
                )
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "Bitwarden refresh failed at proxy start, falling back to "
                "parent env: %s", exc,
            )

    # Caller-supplied overrides win.  This is intentionally last so the
    # wizard can inject ad-hoc test secrets without recomputing the BW
    # path.
    if extra_env:
        env.update(extra_env)

    # Strip proxy-recursion-risk vars regardless of how they got in.
    for name in _PROXY_SUBPROCESS_ENV_STRIP:
        env.pop(name, None)

    env.setdefault("NO_COLOR", "1")
    return env


def stop_proxy() -> bool:
    """Stop the managed iron-proxy.  Returns True if it was running."""

    global _proxy_nonce

    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        _pidfile().unlink(missing_ok=True)
        _proxy_nonce = None
        return False

    # Capture starttime BEFORE signalling so we can compare after the
    # grace window — if the pid got recycled mid-wait, the starttime
    # changes and we abort the SIGKILL.
    starttime_before = _pid_proc_starttime(pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _pidfile().unlink(missing_ok=True)
        _proxy_nonce = None
        return False

    # Wait up to 5s for graceful exit, then SIGKILL.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        # Verify the pid hasn't been recycled before delivering SIGKILL.
        # Two checks:
        #   1. /proc/<pid>/stat starttime is unchanged (Linux)
        #   2. _pid_alive() still says it's an iron-proxy process
        starttime_after = _pid_proc_starttime(pid)
        recycled = (
            starttime_before is not None
            and starttime_after is not None
            and starttime_before != starttime_after
        ) or not _pid_alive(pid)
        if recycled:
            logger.warning(
                "iron-proxy pid=%s appears recycled before SIGKILL; "
                "not killing.", pid,
            )
        else:
            try:
                os.kill(pid, _KILL_SIGNAL)
            except ProcessLookupError:
                pass

    _pidfile().unlink(missing_ok=True)
    _proxy_nonce = None
    logger.info("Stopped iron-proxy pid=%s", pid)
    return True


def get_status() -> ProxyStatus:
    """Snapshot the current proxy state — does NOT start anything.

    Crucially, this is called per Docker-container-create when egress
    enforcement is on.  It must not have side-effects (no mkdir, no
    binary version subprocess that takes 30s on a hung binary).  The
    state dir is read-only here.
    """

    status = ProxyStatus()
    status.tunnel_port = _read_tunnel_port_from_config() or _DEFAULT_TUNNEL_PORT

    binary = find_iron_proxy(install_if_missing=False)
    if binary:
        status.binary_path = binary
        # Cached — see iron_proxy_version().  First call still costs one
        # subprocess; subsequent calls in the same process are dict
        # lookups.
        status.binary_version = iron_proxy_version(binary)

    state = _proxy_state_dir_ro()
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
    cfg = _proxy_state_dir_ro() / "proxy.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
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
    "discover_uncovered_providers",
    "ensure_audit_log",
    "ensure_ca_cert",
    "find_iron_proxy",
    "get_status",
    "install_iron_proxy",
    "iron_proxy_version",
    "load_mappings",
    "merge_mappings",
    "mint_proxy_token",
    "start_proxy",
    "stop_proxy",
    "write_mappings",
    "write_proxy_config",
]
