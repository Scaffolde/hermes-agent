"""Hermetic tests for the iron-proxy egress integration.

Covers the pure-function surface (token mint, mapping discovery, config build,
config + mappings I/O), the binary install path (HTTP downloads + tar
extraction + checksum verification fully mocked), the subprocess lifecycle
(spawn / PID / pid_alive / stop, with subprocess.Popen mocked), and the
docker backend's egress arg builder.

Live network and the real ``iron-proxy`` binary are NEVER touched.  See
``tests/test_iron_proxy_e2e.py`` (gated behind a marker) for the real-binary
smoke test.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from agent.proxy_sources import iron_proxy as ip


# ---------------------------------------------------------------------------
# Per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so install paths don't touch the real $HOME."""

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Make sure no stale provider keys influence discovery.
    for key in list(os.environ):
        if key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    return home


# ---------------------------------------------------------------------------
# Token mint + mapping discovery
# ---------------------------------------------------------------------------


def test_mint_proxy_token_has_prefix_and_length():
    t = ip.mint_proxy_token("alpha")
    assert t.startswith("alpha-")
    assert len(t) >= len("alpha-") + 32


def test_mint_proxy_token_is_random():
    a = ip.mint_proxy_token("x")
    b = ip.mint_proxy_token("x")
    assert a != b


def test_discover_provider_mappings_from_env(hermes_home, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real-2")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    ms = ip.discover_provider_mappings()
    names = [m.real_env_name for m in ms]
    assert "OPENROUTER_API_KEY" in names
    assert "OPENAI_API_KEY" in names
    assert "MISTRAL_API_KEY" not in names


def test_discover_provider_mappings_explicit_names(hermes_home):
    ms = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY", "GROQ_API_KEY", "UNKNOWN_KEY"]
    )
    names = {m.real_env_name for m in ms}
    assert names == {"OPENROUTER_API_KEY", "GROQ_API_KEY"}
    # Unknown providers (no entry in _BEARER_PROVIDERS) are skipped, not warned.


def test_discover_provider_mappings_empty(hermes_home):
    ms = ip.discover_provider_mappings(available_env_names=[])
    assert ms == []


# ---------------------------------------------------------------------------
# Config / mapping serialization
# ---------------------------------------------------------------------------


def _sample_mapping(env_name: str = "OPENROUTER_API_KEY") -> ip.TokenMapping:
    return ip.TokenMapping(
        proxy_token=ip.mint_proxy_token("test"),
        real_env_name=env_name,
        upstream_hosts=("openrouter.ai", "*.openrouter.ai"),
    )


def test_build_proxy_config_shape():
    m = _sample_mapping()
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=Path("/tmp/ca.crt"),
        ca_key=Path("/tmp/ca.key"),
    )
    # Top-level sections — note `dns` is required by iron-proxy even when
    # we only use the CONNECT tunnel.
    assert set(cfg.keys()) >= {"dns", "proxy", "tls", "transforms", "log"}
    # Transforms in expected order
    assert [t["name"] for t in cfg["transforms"]] == ["allowlist", "secrets"]
    # Allowlist uses `domains:` (iron-proxy schema), not `hosts:`
    domains = cfg["transforms"][0]["config"]["domains"]
    assert "openrouter.ai" in domains
    # Secrets transform encodes our mapping
    rules = cfg["transforms"][1]["config"]["secrets"]
    assert len(rules) == 1
    rule = rules[0]
    # Real secret value is sourced from env at egress time, NOT inlined.
    assert rule["source"] == {"type": "env", "var": "OPENROUTER_API_KEY"}
    # The proxy token is the replacement target.
    assert rule["replace"]["proxy_value"] == m.proxy_token
    assert "Authorization" in rule["replace"]["match_headers"]
    # Rules list contains one entry per upstream host.
    rule_hosts = {r["host"] for r in rule["rules"]}
    assert rule_hosts == set(m.upstream_hosts)
    # TLS section names the CA paths
    assert cfg["tls"]["ca_cert"] == "/tmp/ca.crt"


def test_build_proxy_config_custom_allowed_hosts():
    m = _sample_mapping("OPENAI_API_KEY")
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=Path("/tmp/ca.crt"),
        ca_key=Path("/tmp/ca.key"),
        allowed_hosts=["only.example.com"],
    )
    domains = cfg["transforms"][0]["config"]["domains"]
    # Custom allowed_hosts wins as the base; mapping's hosts get appended.
    assert "only.example.com" in domains
    assert "openrouter.ai" in domains  # comes from the mapping


def test_write_and_load_mappings_roundtrip(hermes_home):
    ms = [_sample_mapping("OPENROUTER_API_KEY"), _sample_mapping("OPENAI_API_KEY")]
    path = ip.write_mappings(ms)
    assert path.exists()
    loaded = ip.load_mappings()
    assert len(loaded) == 2
    assert {m.real_env_name for m in loaded} == {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}
    # Tokens preserved
    assert loaded[0].proxy_token == ms[0].proxy_token


def test_load_mappings_handles_missing_file(hermes_home):
    assert ip.load_mappings() == []


def test_load_mappings_handles_corrupt_json(hermes_home):
    state = ip._proxy_state_dir()
    (state / "mappings.json").write_text("{not json", encoding="utf-8")
    assert ip.load_mappings() == []


def test_write_proxy_config_serializes_yaml(hermes_home):
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=Path("/tmp/ca.crt"),
        ca_key=Path("/tmp/ca.key"),
    )
    out = ip.write_proxy_config(cfg)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "tunnel_listen" in text
    assert "ca_cert: /tmp/ca.crt" in text


# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------


def test_find_iron_proxy_returns_none_when_missing(hermes_home, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert ip.find_iron_proxy(install_if_missing=False) is None


def test_find_iron_proxy_returns_managed_first(hermes_home, monkeypatch):
    managed = ip._hermes_bin_dir() / ip._platform_binary_name()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_bytes(b"#!/bin/sh\necho ok\n")
    managed.chmod(0o755)
    # Even with a system one on PATH, the managed copy should win.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/iron-proxy")
    assert ip.find_iron_proxy() == managed


def _make_fake_tar(binary_name: str, payload: bytes = b"#!/bin/sh\necho ok\n") -> bytes:
    """Build a tar.gz with one file at the root, named ``binary_name``."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=binary_name)
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_install_iron_proxy_verifies_checksum_and_extracts(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())
    import hashlib

    expected_sha = hashlib.sha256(fake_payload).hexdigest()
    asset_name = ip._platform_asset_name()
    checksum_text = f"{expected_sha}  {asset_name}\nffff  other-asset.tar.gz\n"

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text(checksum_text)
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    target = ip.install_iron_proxy()
    assert target.exists()
    assert target.read_bytes() == b"#!/bin/sh\necho ok\n"
    # Executable bit is set
    assert os.access(target, os.X_OK)


def test_install_iron_proxy_rejects_bad_checksum(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())
    asset_name = ip._platform_asset_name()
    bad_text = f"deadbeef  {asset_name}\n"

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text(bad_text)
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        ip.install_iron_proxy()


def test_install_iron_proxy_rejects_missing_checksum_entry(hermes_home, monkeypatch):
    fake_payload = _make_fake_tar(ip._platform_binary_name())

    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(ip._IRON_PROXY_CHECKSUM_NAME):
            dest.write_text("aaaa  some-other-file.tar.gz\n")
        else:
            dest.write_bytes(fake_payload)

    monkeypatch.setattr(ip, "_http_download", fake_download)
    with pytest.raises(RuntimeError, match="No checksum entry"):
        ip.install_iron_proxy()


def test_pick_tar_member_rejects_path_traversal():
    """A malicious tar that escapes via '..' must be refused."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="../iron-proxy")
        info.size = 1
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(b"x"))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        with pytest.raises(RuntimeError, match="Could not find iron-proxy"):
            ip._pick_tar_member(tf, "iron-proxy")


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def test_get_status_when_nothing_configured(hermes_home):
    status = ip.get_status()
    assert status.binary_path is None
    assert status.config_path is None
    assert status.ca_cert_path is None
    assert status.pid is None
    assert status.listening is False
    assert not status.installed
    assert not status.configured


def test_get_status_with_config_present(hermes_home, monkeypatch):
    # Materialize binary, config, and ca cert.
    bin_path = ip._hermes_bin_dir() / ip._platform_binary_name()
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(b"")
    bin_path.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake")
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt",
        ca_key=state / "ca.key",
        tunnel_port=9999,
    )
    ip.write_proxy_config(cfg)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "iron-proxy v0.0.0-test")

    status = ip.get_status()
    assert status.installed
    assert status.configured
    assert status.tunnel_port == 9999
    assert "test" in (status.binary_version or "")


def test_stop_proxy_handles_missing_pidfile(hermes_home):
    # No pidfile → stop returns False, doesn't raise.
    assert ip.stop_proxy() is False


def test_stop_proxy_cleans_stale_pidfile(hermes_home, monkeypatch):
    pid_file = ip._proxy_state_dir() / "iron-proxy.pid"
    pid_file.write_text("999999999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: False)
    assert ip.stop_proxy() is False
    assert not pid_file.exists()


def test_start_proxy_refuses_without_binary(hermes_home, monkeypatch):
    # No binary, auto_install fails → RuntimeError surfaces.
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: None)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")
    with pytest.raises(RuntimeError, match="binary not available"):
        ip.start_proxy()


def test_start_proxy_refuses_without_config(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    with pytest.raises(RuntimeError, match="config not found"):
        ip.start_proxy()


def test_start_proxy_writes_pidfile_when_alive(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    monkeypatch.setattr(ip, "_STARTUP_GRACE_SECONDS", 0)

    # Pre-stub everything start_proxy's get_status() call will touch — it
    # runs INSIDE start_proxy, so by the time Popen is mocked these have
    # to already be hermetic.
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: False)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "iron-proxy test")

    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = None  # still alive

    with patch("subprocess.Popen", lambda *a, **k: fake_proc):
        status = ip.start_proxy()
    assert (state / "iron-proxy.pid").read_text() == "4242"
    assert status.pid == 4242


def test_start_proxy_raises_when_immediate_exit(hermes_home, monkeypatch):
    binary = ip._hermes_bin_dir() / ip._platform_binary_name()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")
    binary.chmod(0o755)
    state = ip._proxy_state_dir()
    (state / "proxy.yaml").write_text("proxy: {}")
    (state / "iron-proxy.log").write_text("bind: address already in use\n")

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kwargs: binary)
    monkeypatch.setattr(ip, "_STARTUP_GRACE_SECONDS", 0)

    fake_proc = MagicMock()
    fake_proc.pid = 5151
    fake_proc.poll.return_value = 1  # exited immediately
    fake_proc.returncode = 1
    with patch("subprocess.Popen", lambda *a, **k: fake_proc):
        with pytest.raises(RuntimeError, match="exited immediately"):
            ip.start_proxy()


def test_start_proxy_idempotent_when_already_running(hermes_home, monkeypatch):
    state = ip._proxy_state_dir()
    pid_file = state / "iron-proxy.pid"
    pid_file.write_text("12345")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    # Materialize config so we get past that check (we shouldn't reach it,
    # but if the idempotent path regresses we want a clean failure mode).
    (state / "proxy.yaml").write_text("proxy: {}")
    # Sentinel: subprocess.Popen must NOT be called.
    with patch("subprocess.Popen", lambda *a, **k: pytest.fail("should not spawn")):
        status = ip.start_proxy()
    # Should return without spawning anything.
    assert status is not None


# ---------------------------------------------------------------------------
# Docker integration
# ---------------------------------------------------------------------------


def test_docker_egress_args_empty_when_disabled(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker

    # Default config has proxy.enabled=False; helper should return all empties.
    vol, env, host = _egress_proxy_args_for_docker()
    assert vol == []
    assert env == {}
    assert host == []


def test_docker_egress_args_when_enabled_but_unconfigured_raises(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    # No proxy.yaml exists → enforce_on_docker should raise.
    with pytest.raises(RuntimeError, match="not configured"):
        _egress_proxy_args_for_docker()


def test_docker_egress_args_when_unconfigured_no_enforce(hermes_home, monkeypatch):
    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = False
    save_config(cfg)

    # Without enforcement, missing config returns empties (warning only).
    vol, env, host = _egress_proxy_args_for_docker()
    assert vol == []
    assert env == {}
    assert host == []


def test_docker_egress_args_full_path(hermes_home, monkeypatch):
    """Wire up everything (config, CA, mappings, fake running proxy) and
    verify the docker helper emits the right mounts and env."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    # Materialize config, CA, mappings.
    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    mapping = _sample_mapping("OPENROUTER_API_KEY")
    proxy_cfg = ip.build_proxy_config(
        mappings=[mapping], ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([mapping])

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    # Fake running proxy.
    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    vol, env, host = _egress_proxy_args_for_docker()
    # CA mount present and in -v form
    assert "-v" in vol
    assert any("hermes-egress-ca.crt" in arg for arg in vol)
    # Env contains both casings of HTTPS_PROXY and the CA env vars
    assert env["HTTPS_PROXY"].endswith(":9090")
    assert env["https_proxy"] == env["HTTPS_PROXY"]
    assert env["REQUESTS_CA_BUNDLE"].endswith("hermes-egress-ca.crt")
    assert env["NODE_EXTRA_CA_CERTS"] == env["REQUESTS_CA_BUNDLE"]
    # NO_PROXY excludes loopback
    assert "127.0.0.1" in env["NO_PROXY"]
    # Per-mapping proxy token surfaced
    assert env["HERMES_PROXY_TOKEN_OPENROUTER_API_KEY"] == mapping.proxy_token
    # Linux host-gateway mapping
    assert host == ["--add-host", "host.docker.internal:host-gateway"]


# ---------------------------------------------------------------------------
# Platform asset name resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,machine,expected_substring",
    [
        ("Linux", "x86_64", "linux_amd64"),
        ("Linux", "aarch64", "linux_arm64"),
        ("Darwin", "arm64", "darwin_arm64"),
        ("Darwin", "x86_64", "darwin_amd64"),
    ],
)
def test_platform_asset_name(monkeypatch, system, machine, expected_substring):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert expected_substring in ip._platform_asset_name()


def test_platform_asset_name_rejects_windows(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="does not ship native Windows"):
        ip._platform_asset_name()
