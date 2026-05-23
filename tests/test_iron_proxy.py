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


def test_build_proxy_config_shape(tmp_path):
    m = _sample_mapping()
    ca_crt = tmp_path / "ca.crt"
    ca_key = tmp_path / "ca.key"
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=ca_crt,
        ca_key=ca_key,
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
    assert cfg["tls"]["ca_cert"] == str(ca_crt)


def test_build_proxy_config_custom_allowed_hosts(tmp_path):
    m = _sample_mapping("OPENAI_API_KEY")
    cfg = ip.build_proxy_config(
        mappings=[m],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        allowed_hosts=["custom-host.test"],
    )
    domains = cfg["transforms"][0]["config"]["domains"]
    # Custom allowed_hosts wins as the base; mapping's hosts get appended.
    assert "custom-host.test" in domains
    assert "openrouter.ai" in domains  # comes from the mapping


# ---------------------------------------------------------------------------
# Default SSRF deny list (regression: docs promise cloud metadata is denied)
# ---------------------------------------------------------------------------


def test_default_deny_cidrs_present_when_unspecified(tmp_path):
    """build_proxy_config must emit the default deny list when the caller
    passes nothing.  The IMDS subnet (169.254.0.0/16) MUST be in the result
    or the docs claim that ``upstream_deny_cidrs`` refuses cloud metadata
    is a lie."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
    )
    deny = cfg["proxy"]["upstream_deny_cidrs"]
    assert "169.254.0.0/16" in deny  # IMDS
    assert "127.0.0.0/8" in deny      # loopback v4
    assert "::1/128" in deny           # loopback v6
    assert "10.0.0.0/8" in deny        # RFC1918
    assert "172.16.0.0/12" in deny     # RFC1918
    assert "192.168.0.0/16" in deny    # RFC1918


def test_explicit_empty_deny_cidrs_disables_default(tmp_path):
    """Explicit ``[]`` opts out of the default deny list — needed by
    hermetic tests that want to talk to a loopback upstream."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        upstream_deny_cidrs=[],
    )
    assert cfg["proxy"]["upstream_deny_cidrs"] == []


def test_wizard_rendered_yaml_contains_deny_list(hermes_home, tmp_path):
    """End-to-end: cmd_setup writes proxy.yaml; the rendered file must
    contain the deny list because the wizard now passes the operator's
    config-level setting (None → default) through to build_proxy_config."""

    # Simulate the wizard's call shape (matches proxy_cli.cmd_setup).
    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_yaml = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt",
        ca_key=state / "ca.key",
        # The wizard passes ``upstream_deny_cidrs`` from the config; when
        # the operator hasn't set anything, that's None and we get the
        # safe default below.
        upstream_deny_cidrs=None,
    )
    out = ip.write_proxy_config(proxy_yaml)
    text = out.read_text(encoding="utf-8")
    assert "169.254.0.0/16" in text


# ---------------------------------------------------------------------------
# Bind policy (regression: must not bind 0.0.0.0)
# ---------------------------------------------------------------------------


def test_default_bind_is_loopback_not_zero_zero(tmp_path):
    """``http_listen`` must NOT be ``0.0.0.0:PORT`` or ``:PORT`` (latter is
    INADDR_ANY).  Loopback only by default; the docker bridge bind is
    optional and added in addition, never instead."""

    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=12345,
        http_listen=["127.0.0.1:12345"],  # explicit so test is deterministic
    )
    primary = cfg["proxy"]["http_listen"]
    listens = cfg["proxy"]["http_listens"]
    assert primary == "127.0.0.1:12345"
    assert listens == ["127.0.0.1:12345"]
    # Sentinel: confirm we didn't accidentally serialize a bare-port form
    # like ":12345" anywhere in the listen list (that's INADDR_ANY).
    for entry in listens:
        assert not entry.startswith(":")
        assert "0.0.0.0" not in entry


def test_default_bind_includes_docker_bridge_on_linux(tmp_path, monkeypatch):
    """When http_listen isn't passed AND we're on Linux AND a docker
    bridge IP is detected, we should bind that bridge IP in addition to
    loopback so containers reach the proxy via host-gateway."""

    monkeypatch.setattr(ip.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ip, "_detect_docker_bridge_ip", lambda: "172.17.0.1")
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        tunnel_port=9090,
    )
    assert "127.0.0.1:9090" in cfg["proxy"]["http_listens"]
    assert "172.17.0.1:9090" in cfg["proxy"]["http_listens"]


# ---------------------------------------------------------------------------
# audit_log wiring (regression: parameter was accepted but never used)
# ---------------------------------------------------------------------------


def test_audit_log_path_lands_in_yaml(tmp_path):
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        audit_log=tmp_path / "audit.log",
    )
    assert cfg["log"]["audit_path"] == str(tmp_path / "audit.log")


def test_audit_log_omitted_when_caller_passes_none(tmp_path):
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=tmp_path / "ca.crt",
        ca_key=tmp_path / "ca.key",
        audit_log=None,
    )
    assert "audit_path" not in cfg["log"]


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


def test_write_proxy_config_serializes_yaml(hermes_home, tmp_path):
    ca_crt = tmp_path / "ca.crt"
    ca_key = tmp_path / "ca.key"
    cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca_crt,
        ca_key=ca_key,
    )
    out = ip.write_proxy_config(cfg)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "tunnel_listen" in text
    assert f"ca_cert: {ca_crt}" in text


# ---------------------------------------------------------------------------
# Token-preservation on re-setup (regression: clobbered live sandboxes)
# ---------------------------------------------------------------------------


def test_merge_mappings_preserves_existing_tokens():
    """Re-running setup must not invalidate tokens baked into already-
    running sandboxes.  ``merge_mappings`` keeps the prior token for any
    provider that's in both lists."""

    existing = [
        ip.TokenMapping(
            proxy_token="hermes-proxy-original-12345",
            real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered)
    by_name = {m.real_env_name: m for m in merged}
    # Original token preserved.
    assert by_name["OPENROUTER_API_KEY"].proxy_token == "hermes-proxy-original-12345"
    # New provider got a fresh token.
    assert by_name["OPENAI_API_KEY"].proxy_token != "hermes-proxy-original-12345"
    # Both providers in the result.
    assert set(by_name) == {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}


def test_merge_mappings_drops_providers_removed_from_env():
    """When a provider is in `existing` but not in `discovered`, it must
    be dropped from the result — the operator removed the env var."""

    existing = [
        ip.TokenMapping(
            proxy_token="stale", real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENAI_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered)
    names = {m.real_env_name for m in merged}
    assert names == {"OPENAI_API_KEY"}


def test_merge_mappings_rotate_mints_fresh_tokens():
    """``rotate=True`` rolls every token regardless of overlap.  The
    --rotate-tokens flag uses this."""

    existing = [
        ip.TokenMapping(
            proxy_token="hermes-proxy-original-12345",
            real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ]
    discovered = ip.discover_provider_mappings(
        available_env_names=["OPENROUTER_API_KEY"]
    )
    merged = ip.merge_mappings(existing=existing, discovered=discovered, rotate=True)
    assert merged[0].proxy_token != "hermes-proxy-original-12345"


# ---------------------------------------------------------------------------
# Uncovered provider detection (regression: non-bearer providers bypass)
# ---------------------------------------------------------------------------


def test_uncovered_providers_detects_anthropic_aws(hermes_home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    uncovered = ip.discover_uncovered_providers()
    assert "ANTHROPIC_API_KEY" in uncovered
    assert "AWS_ACCESS_KEY_ID" in uncovered


def test_uncovered_providers_explicit_names_empty():
    assert ip.discover_uncovered_providers(available_env_names=[]) == []


def test_uncovered_providers_skips_bearer_providers(hermes_home, monkeypatch):
    """OPENROUTER_API_KEY etc. are bearer providers — they should NOT
    appear in the uncovered list."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    uncovered = ip.discover_uncovered_providers()
    assert "OPENROUTER_API_KEY" not in uncovered


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


# ---------------------------------------------------------------------------
# Subprocess env minimization (regression: host secrets leaked to proxy)
# ---------------------------------------------------------------------------


def test_subprocess_env_strips_unrelated_secrets(hermes_home, monkeypatch):
    """``_build_proxy_subprocess_env`` must NOT carry every host secret
    over to the proxy.  /proc/<pid>/environ on the proxy would otherwise
    expose all of them to same-uid local processes."""

    # Unrelated env vars that should NOT propagate.
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://very-private")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-very-secret")
    # Provider keys that ARE in load_mappings should propagate.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    ip.write_mappings([_sample_mapping("OPENROUTER_API_KEY")])

    env = ip._build_proxy_subprocess_env()
    assert "MY_PRIVATE_TOKEN" not in env
    assert "DATABASE_URL" not in env
    assert "SLACK_BOT_TOKEN" not in env
    assert env.get("OPENROUTER_API_KEY") == "sk-or-real"


def test_subprocess_env_strips_proxy_recursion_vars(hermes_home, monkeypatch):
    """HTTPS_PROXY etc. in the parent env would otherwise recurse iron-proxy
    through itself (or send its traffic through a corporate proxy)."""

    monkeypatch.setenv("HTTPS_PROXY", "http://corporate:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://corporate:3128")
    monkeypatch.setenv("ALL_PROXY", "socks5://corporate:1080")
    env = ip._build_proxy_subprocess_env()
    assert "HTTPS_PROXY" not in env
    assert "https_proxy" not in env
    assert "HTTP_PROXY" not in env
    assert "ALL_PROXY" not in env


def test_subprocess_env_keeps_infrastructure_vars(hermes_home, monkeypatch):
    """PATH / HOME / locale must propagate or the child can't even find
    its libs."""

    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = ip._build_proxy_subprocess_env()
    assert env.get("PATH") == "/usr/local/bin:/usr/bin"
    assert env.get("HOME") == "/home/test"
    assert env.get("LANG") == "C.UTF-8"


# ---------------------------------------------------------------------------
# CA generation TOCTOU (regression: 0o600 only set AFTER copy)
# ---------------------------------------------------------------------------


def test_ca_key_created_with_0o600(hermes_home, monkeypatch):
    """The CA private key must NEVER exist on disk with default umask
    permissions, even transiently.  Fix: open with explicit mode=0o600
    so the very first byte is written under tight perms."""

    # ensure_ca_cert shells out to openssl; mock the subprocess.run calls
    # so we don't need openssl on the test host AND don't depend on its
    # output format.
    def fake_run(args, **kwargs):
        # First call: genrsa → -out is at args[-2]
        if args[1] == "genrsa":
            out = args[-2]
            Path(out).write_bytes(b"-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n")
        elif args[1] == "req":
            # Find -out path
            i = args.index("-out")
            Path(args[i + 1]).write_bytes(b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        result = MagicMock()
        result.returncode = 0
        return result

    monkeypatch.setattr(ip.shutil, "which", lambda name: "/usr/bin/openssl" if name == "openssl" else None)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)

    ca_crt, ca_key = ip.ensure_ca_cert()
    assert ca_key.exists()
    mode = ca_key.stat().st_mode & 0o777
    assert mode == 0o600, f"CA key has perms {oct(mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Audit log permissions (regression: depended on umask)
# ---------------------------------------------------------------------------


def test_ensure_audit_log_creates_with_0o600(hermes_home, tmp_path):
    audit = tmp_path / "audit.log"
    ip.ensure_audit_log(audit)
    assert audit.exists()
    mode = audit.stat().st_mode & 0o777
    assert mode == 0o600


def test_ensure_audit_log_tightens_existing_perms(hermes_home, tmp_path):
    audit = tmp_path / "audit.log"
    audit.write_text("preexisting content\n")
    os.chmod(audit, 0o644)
    ip.ensure_audit_log(audit)
    mode = audit.stat().st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# State dir hardening (regression: world-traversable on multi-user hosts)
# ---------------------------------------------------------------------------


def test_proxy_state_dir_is_0o700(hermes_home):
    state = ip._proxy_state_dir()
    mode = state.stat().st_mode & 0o777
    assert mode == 0o700


def test_proxy_state_dir_ro_does_not_create(hermes_home):
    """_proxy_state_dir_ro is for read-only callers — it must NOT
    materialize the dir.  Pure-status code paths shouldn't have the
    side-effect of creating ~/.hermes/proxy/."""

    # Sanity: rw path creates it.
    rw = ip._proxy_state_dir()
    assert rw.exists()
    # Remove it and confirm the ro path doesn't recreate.
    import shutil as _shutil
    _shutil.rmtree(str(rw))
    assert not rw.exists()
    ro = ip._proxy_state_dir_ro()
    assert not ro.exists()
    # The path string is the same as the rw one.
    assert ro == rw


# ---------------------------------------------------------------------------
# Mappings clobber refused when corrupt (regression: silent 403s)
# ---------------------------------------------------------------------------


def test_docker_egress_args_raises_on_empty_mappings(hermes_home, monkeypatch):
    """If mappings.json is missing / corrupt / empty AND
    enforce_on_docker is true, refuse to start the sandbox rather than
    silently mounting an unusable proxy config."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    state = ip._proxy_state_dir()
    (state / "ca.crt").write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=state / "ca.crt", ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    # Note: we deliberately do NOT write mappings.json — that's the
    # bug-class this test guards against.

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    with pytest.raises(RuntimeError, match="mappings.json is empty or"):
        _egress_proxy_args_for_docker()


# ---------------------------------------------------------------------------
# CA missing → enforce_on_docker semantics (regression: silent fail-open)
# ---------------------------------------------------------------------------


def test_docker_egress_args_raises_when_ca_vanishes(hermes_home, monkeypatch):
    """status.configured was True at check time but the CA file
    disappeared between then and now (e.g. operator manually deleted
    ~/.hermes/proxy/ca.crt).  enforce_on_docker=True must refuse."""

    from tools.environments.docker import _egress_proxy_args_for_docker
    from hermes_cli.config import load_config, save_config

    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([_sample_mapping()])

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)

    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    # Build a fake status: configured=True (because both path fields are
    # set), but ca_cert_path.exists() is False — simulating the race
    # where the CA file vanished between get_status() and the
    # exists() recheck inside _egress_proxy_args_for_docker.
    fake_status = ip.ProxyStatus(
        binary_path=state / "fake-bin",  # truthy
        config_path=state / "proxy.yaml",
        ca_cert_path=state / "missing-ca.crt",  # points at nonexistent path
        pid=99999,
        listening=True,
        tunnel_port=9090,
    )
    # ProxyStatus.configured returns True iff config_path AND ca_cert_path
    # both exist.  We need configured=True but the second exists() check
    # in docker.py to return False — force that by writing a placeholder
    # config_path that exists and pointing ca_cert_path at a missing file.
    (state / "proxy.yaml").write_text("# fake")
    # ProxyStatus.configured: config_path.exists() and ca_cert_path.exists().
    # Make ca_cert_path .exists() True for the configured check but the
    # explicit .exists() recheck path in docker.py reads the same Path,
    # which is missing — so we wrap.
    class _CAStub:
        """Path-like that toggles .exists() so configured=True but the
        defensive recheck in docker.py returns False."""
        _calls = 0
        def __init__(self, real: Path):
            self._real = real
        def __str__(self):
            return str(self._real)
        @property
        def parent(self):
            return self._real.parent
        def exists(self):
            type(self)._calls += 1
            # First call: configured property check → say yes.
            # Second call: docker.py defensive recheck → say no.
            return type(self)._calls == 1
    fake_status.ca_cert_path = _CAStub(state / "missing-ca.crt")  # type: ignore[assignment]
    monkeypatch.setattr(ip, "get_status", lambda: fake_status)

    with pytest.raises(RuntimeError, match="CA cert vanished"):
        _egress_proxy_args_for_docker()


# ---------------------------------------------------------------------------
# Docker env collision detection (regression: docker_env silently bypassed proxy)
# ---------------------------------------------------------------------------


def test_docker_env_collision_with_proxy_raises_when_enforce(hermes_home, monkeypatch):
    """Setting docker_env: {HTTPS_PROXY: ''} in config.yaml with
    enforce_on_docker=true must fail-loud rather than silently inverting
    the egress isolation."""

    from tools.environments.docker import DockerEnvironment
    from hermes_cli.config import load_config, save_config

    # Set up a fully-running proxy.
    state = ip._proxy_state_dir()
    ca = state / "ca.crt"
    ca.write_text("fake-ca")
    (state / "ca.key").write_text("fake-key")
    proxy_cfg = ip.build_proxy_config(
        mappings=[_sample_mapping()],
        ca_cert=ca, ca_key=state / "ca.key", tunnel_port=9090,
    )
    ip.write_proxy_config(proxy_cfg)
    ip.write_mappings([_sample_mapping()])
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["enforce_on_docker"] = True
    save_config(cfg)
    (state / "iron-proxy.pid").write_text("99999")
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    # Mock the docker availability check so we never shell out.
    monkeypatch.setattr(
        "tools.environments.docker._ensure_docker_available", lambda: None,
    )
    # Mock find_docker so the resolved docker exe isn't probed.
    monkeypatch.setattr(
        "tools.environments.docker.find_docker", lambda: "/bin/true",
    )
    # Mock subprocess.run so we don't actually run `docker run`.  We
    # only need the constructor to get past the env merge logic.
    monkeypatch.setattr(
        "tools.environments.docker.subprocess.run",
        lambda *a, **k: MagicMock(stdout="abc123\n", returncode=0),
    )
    # init_session is the second outbound subprocess we don't care about.
    monkeypatch.setattr(
        "tools.environments.docker.DockerEnvironment.init_session",
        lambda self: None,
    )

    # The collision: user sets HTTPS_PROXY to empty string in docker_env.
    with pytest.raises(RuntimeError, match="overrides egress-proxy variables"):
        DockerEnvironment(
            image="busybox",
            env={"HTTPS_PROXY": ""},  # the collision
        )
