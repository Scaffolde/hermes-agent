"""Tests for LSP client eviction — idle timeout and LRU cap (SCA-4389).

Why this file exists at all:

``LSPService`` accepted an ``idle_timeout``, stored it on the instance,
and never read it again.  ``DEFAULT_IDLE_TIMEOUT``'s comment claimed
"servers idle for >10min get reaped".  Nothing reaped anything.  On a
16 GiB host that produced 13 live ``typescript-language-server`` trees
holding ~16 GiB, which pushed the box into swap, which pushed free
disk under the CI runner's admission floor and took CI offline.

So the bar here is specifically *not* "the policy function returns the
right list".  A cache with an eviction path that never executes is the
exact ``--dry-run`` false-green this issue was filed about.  The two
headline tests below drive **real spawned clients** through the real
request path and assert the process is actually gone, and each carries
a **positive control** — the same scenario with the bound disabled,
proving the server survives when it should.  A green with no
demonstrated red would prove nothing.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

from agent.lsp import eventlog, workspace
from agent.lsp.manager import (
    DEFAULT_IDLE_TIMEOUT,
    LSPService,
    default_max_servers,
)
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock LSP as ``pyright`` and neutralise cwd anchoring.

    ``resolve_workspace_for_file`` prefers the cwd's git worktree over
    the file's own.  These tests need **distinct** workspace roots per
    file, so the cwd is parked in a non-git directory to force the
    per-file anchor.
    """
    home = tmp_path / "not-a-repo"
    home.mkdir()
    monkeypatch.chdir(str(home))

    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": "errors"},
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock pyright",
    )
    workspace.clear_cache()
    eventlog.reset_announce_caches()
    yield tmp_path
    SERVERS[target_index] = original
    workspace.clear_cache()


def _make_repo(root: Path, name: str) -> Path:
    """A minimal git worktree with one Python file, returning the file."""
    repo = root / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    f = repo / "x.py"
    f.write_text("print('hi')\n")
    return f


def _service(**kw) -> LSPService:
    kw.setdefault("enabled", True)
    kw.setdefault("wait_mode", "document")
    kw.setdefault("wait_timeout", 3.0)
    kw.setdefault("install_strategy", "manual")
    return LSPService(**kw)


def _roots(svc: LSPService):
    return {key[1] for key in svc._clients}


# ---------------------------------------------------------------------------
# AC1 — idle timeout actually shuts a server down
# ---------------------------------------------------------------------------


def test_idle_server_is_evicted_and_survives_when_bound_disabled(mock_pyright):
    """An idle server past the timeout is shut down for real.

    Both halves run the *identical* scenario; only ``idle_timeout``
    differs.  The second half is the positive control — with the bound
    off, repo A must still be alive, so the first half's eviction is
    attributable to the timeout and not to some unrelated teardown.
    """
    a = _make_repo(mock_pyright, "repo_a")
    b = _make_repo(mock_pyright, "repo_b")

    # --- bound ON: A goes idle, touching B evicts it -----------------
    svc = _service(idle_timeout=0.25, max_servers=0)  # cap disabled
    try:
        svc.snapshot_baseline(str(a))
        assert str(a.parent) in _roots(svc), "A never spawned; test proves nothing"
        client_a = svc._clients[("pyright", str(a.parent))]

        time.sleep(0.4)  # A is now idle past the timeout
        svc.snapshot_baseline(str(b))  # request path is the clock

        assert str(a.parent) not in _roots(svc), "idle server was NOT evicted"
        assert str(b.parent) in _roots(svc), "the requested server must survive"
        assert not client_a.is_running, "evicted client's process is still alive"
    finally:
        svc.shutdown()

    # --- bound OFF (positive control): A survives the same sequence ---
    workspace.clear_cache()
    svc2 = _service(idle_timeout=0, max_servers=0)
    try:
        svc2.snapshot_baseline(str(a))
        time.sleep(0.4)
        svc2.snapshot_baseline(str(b))
        assert str(a.parent) in _roots(svc2), (
            "control failed: A died with eviction disabled, so the first "
            "half's result cannot be attributed to the idle timeout"
        )
    finally:
        svc2.shutdown()


# ---------------------------------------------------------------------------
# AC2 — LRU cap bounds the population independently of idleness
# ---------------------------------------------------------------------------


def test_cap_plus_one_evicts_least_recently_used(mock_pyright):
    """The (cap+1)-th root evicts the LRU, not an arbitrary victim.

    Every server here is freshly used, so the idle timeout cannot be
    what fires — this isolates the cap.
    """
    a = _make_repo(mock_pyright, "repo_a")
    b = _make_repo(mock_pyright, "repo_b")
    c = _make_repo(mock_pyright, "repo_c")

    svc = _service(idle_timeout=0, max_servers=2)  # idle bound disabled
    try:
        svc.snapshot_baseline(str(a))
        time.sleep(0.05)
        svc.snapshot_baseline(str(b))
        assert _roots(svc) == {str(a.parent), str(b.parent)}
        client_a = svc._clients[("pyright", str(a.parent))]

        # Re-touch B so A is unambiguously the least-recently-used.
        time.sleep(0.05)
        svc.snapshot_baseline(str(b))

        svc.snapshot_baseline(str(c))  # third root, cap is 2

        assert len(svc._clients) <= 2, f"cap breached: {_roots(svc)}"
        assert str(a.parent) not in _roots(svc), "LRU root was not the victim"
        assert str(b.parent) in _roots(svc), "wrongly evicted the recently-used root"
        assert str(c.parent) in _roots(svc), "the requested root must survive"
        assert not client_a.is_running
    finally:
        svc.shutdown()


def test_population_never_exceeds_cap_across_many_roots(mock_pyright):
    """The steady state of multi-worktree work stays bounded.

    This is the shape of the original defect: N worktrees touched over
    time, each leaving a permanent ~1.3 GiB tenant.
    """
    svc = _service(idle_timeout=0, max_servers=2)
    try:
        for i in range(6):
            f = _make_repo(mock_pyright, f"repo_{i}")
            svc.snapshot_baseline(str(f))
            assert len(svc._clients) <= 2, (
                f"after {i + 1} roots the fleet is {len(svc._clients)}, cap is 2"
            )
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# AC6 — never evict a server mid-request
# ---------------------------------------------------------------------------


def test_inflight_server_is_never_evicted():
    """A draining request protects its server from both bounds."""
    svc = _service(enabled=False, idle_timeout=1, max_servers=1)
    try:
        old = ("pyright", "/repo/old")
        new = ("pyright", "/repo/new")
        svc._clients[old] = object()
        svc._clients[new] = object()
        svc._last_used[old] = 0.0  # ancient: idle AND the LRU
        svc._last_used[new] = time.time()

        # Not in flight -> it is a candidate under both rules.
        assert any(k == old for k, _ in svc._eviction_candidates())

        # In flight -> immune, even though nothing else changed.
        svc._acquire(old)
        assert not any(k == old for k, _ in svc._eviction_candidates()), (
            "an in-flight server was selected for eviction"
        )

        # Released -> the release stamps it as just-used, so it is
        # neither idle nor the LRU any more.  ``new`` becomes the cap
        # victim instead.  This is the interesting direction: finishing
        # a request must count as activity, otherwise a busy server
        # would be evicted the instant it went quiet.
        svc._release(old)
        reasons = {k: r for k, r in svc._eviction_candidates()}
        assert old not in reasons, "a just-released server was still a victim"
        assert new in reasons, "the cap must still bind after the release"
        assert "lru cap" in reasons[new]
    finally:
        svc.shutdown()


def test_protected_key_is_never_its_own_victim():
    """With a cap of 1, the root being requested must not be evicted."""
    svc = _service(enabled=False, idle_timeout=0, max_servers=1)
    try:
        key = ("pyright", "/repo/only")
        svc._clients[key] = object()
        svc._last_used[key] = 0.0
        assert svc._eviction_candidates(protect=key) == []
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# AC3 — bounds are configurable, defaults derived from host memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_bytes,expected",
    [
        (16 * 1024**3, 3),    # Mac Mini that produced this defect
        (128 * 1024**3, 26),  # workstation
        (8 * 1024**3, 1),     # small host still gets a usable floor
        (None, 4),            # undiscoverable -> assume small
    ],
)
def test_default_max_servers_scales_with_host_memory(monkeypatch, total_bytes, expected):
    monkeypatch.setattr(
        "agent.lsp.manager._host_memory_bytes", lambda: total_bytes
    )
    assert default_max_servers() == expected


def test_derived_cap_is_bounded_on_an_enormous_host(monkeypatch):
    """A 2 TiB host must not derive a cap that is not a bound."""
    monkeypatch.setattr(
        "agent.lsp.manager._host_memory_bytes", lambda: 2048 * 1024**3
    )
    assert default_max_servers() == 32


def test_config_plumbs_both_bounds(monkeypatch):
    """``lsp.idle_timeout`` / ``lsp.max_servers`` reach the service."""
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(
        cfg_mod, "load_config",
        lambda *a, **k: {"lsp": {"idle_timeout": 42, "max_servers": 7}},
    )
    svc = LSPService.create_from_config()
    try:
        assert svc is not None
        assert svc._idle_timeout == 42
        assert svc._max_servers == 7
    finally:
        if svc is not None:
            svc.shutdown()


def test_config_defaults_when_absent(monkeypatch):
    """Absent config derives the cap rather than hardcoding one."""
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda *a, **k: {"lsp": {}})
    svc = LSPService.create_from_config()
    try:
        assert svc is not None
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._max_servers == default_max_servers()
    finally:
        if svc is not None:
            svc.shutdown()


def test_malformed_config_values_fall_back_rather_than_crash(monkeypatch):
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(
        cfg_mod, "load_config",
        lambda *a, **k: {"lsp": {"idle_timeout": "soon", "max_servers": "lots"}},
    )
    svc = LSPService.create_from_config()
    try:
        assert svc is not None
        assert svc._idle_timeout == DEFAULT_IDLE_TIMEOUT
        assert svc._max_servers == default_max_servers()
    finally:
        if svc is not None:
            svc.shutdown()


# ---------------------------------------------------------------------------
# AC4 — eviction is observable
# ---------------------------------------------------------------------------


def test_eviction_logs_root_and_reason(mock_pyright, caplog):
    """"The cache never evicts" must be falsifiable from the log."""
    a = _make_repo(mock_pyright, "repo_a")
    b = _make_repo(mock_pyright, "repo_b")

    svc = _service(idle_timeout=0.25, max_servers=0)
    try:
        svc.snapshot_baseline(str(a))
        time.sleep(0.4)
        with caplog.at_level(logging.INFO, logger="hermes.lint.lsp"):
            svc.snapshot_baseline(str(b))
    finally:
        svc.shutdown()

    evictions = [r.getMessage() for r in caplog.records if "evicted" in r.getMessage()]
    assert evictions, "eviction happened with no log line"
    assert any(str(a.parent) in m for m in evictions), "log does not name the root"
    assert any("idle" in m for m in evictions), "log does not give the reason"


def test_evicted_root_reannounces_as_active_on_respawn(caplog):
    """A re-spawn after eviction is a new process and must say so.

    ``log_active`` announces INFO once per root and DEBUG thereafter.
    Without clearing that entry on eviction, a respawned server would
    only ever log "reused client" — which would be false.
    """
    eventlog.reset_announce_caches()
    with caplog.at_level(logging.DEBUG, logger="hermes.lint.lsp"):
        eventlog.log_active("pyright", "/repo/a")
        eventlog.log_active("pyright", "/repo/a")  # deduped -> DEBUG
        eventlog.log_evicted("pyright", "/repo/a", "idle 700s >= 600s")
        eventlog.log_active("pyright", "/repo/a")  # must be INFO again

    active_info = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "active for /repo/a" in r.getMessage()
    ]
    assert len(active_info) == 2, (
        f"expected 2 INFO 'active' lines (pre- and post-eviction), got {len(active_info)}"
    )


def test_status_exposes_bounds_and_idleness():
    """``hermes lsp status`` can answer "is the cache evicting?"."""
    svc = _service(enabled=False, idle_timeout=600, max_servers=3)
    try:
        status = svc.get_status()
        assert status["idle_timeout"] == 600
        assert status["max_servers"] == 3
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Regression guard on the defect itself
# ---------------------------------------------------------------------------


def test_idle_timeout_is_actually_read():
    """Pin the exact defect: the bound must not be write-only again.

    Before SCA-4389 ``_idle_timeout`` was assigned in ``__init__`` and
    referenced nowhere else.  This asserts it changes behaviour.
    """
    old = ("pyright", "/repo/old")

    never = _service(enabled=False, idle_timeout=0, max_servers=0)
    strict = _service(enabled=False, idle_timeout=1, max_servers=0)
    try:
        for svc in (never, strict):
            svc._clients[old] = object()
            svc._last_used[old] = 0.0  # ancient

        assert never._eviction_candidates() == [], "idle_timeout=0 must disable the bound"
        assert [k for k, _ in strict._eviction_candidates()] == [old], (
            "idle_timeout is being ignored — the SCA-4389 defect is back"
        )
    finally:
        never.shutdown()
        strict.shutdown()
