"""Service-level orchestration for LSP clients.

The :class:`LSPService` is the bridge between the synchronous
file_operations layer and the async :class:`agent.lsp.client.LSPClient`.

Design choices:

- A **single asyncio event loop** runs in a background thread.  All
  client work happens on that loop.  Synchronous callers from
  ``tools/file_operations.py`` use :meth:`get_diagnostics_sync` to
  open + wait + drain in one blocking call.

- One client per ``(server_id, workspace_root)`` key.  Lazy spawn:
  the first request for a key spawns the client; subsequent requests
  re-use it.

- The client population is **bounded** by two independent rules,
  both applied on the request path (see :meth:`_enforce_bounds`):
  an **idle timeout** evicts a root nobody has asked about for
  ``idle_timeout`` seconds, and an **LRU cap** evicts the
  least-recently-used root once the population would exceed
  ``max_servers``.  The cap's default is derived from host RAM
  (:func:`default_max_servers`), because one language server against
  a large TypeScript project costs ~1.3 GiB and a 16 GiB host cannot
  afford the same fleet as a 128 GiB one.  A server draining an
  in-flight request is never evicted.  Every eviction emits an INFO
  line naming the root and the reason.

  This is deliberately spelled out because it was previously false:
  ``idle_timeout`` was accepted, stored, and never read, while the
  constant that carried it claimed servers "get reaped".  Thirteen
  live servers holding ~16 GiB on a 16 GiB host was the result
  (SCA-4389).  If a future change removes the eviction call, delete
  this paragraph with it — a comment describing a reaper that does
  not run is worse than no comment.

- A **broken-set** records ``(server_id, workspace_root)`` pairs that
  failed to spawn or initialize.  These are never retried for the
  life of the service.  Mirrors OpenCode's design.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
)
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get evicted

# Measured footprint of one ``typescript-language-server`` plus its
# ``tsserver`` child against a scaffolde-ai checkout: 1.2-1.6 GiB
# resident.  1.3 GiB is the middle of that range and is the unit the
# default cap is denominated in.  It is a sizing constant, not a
# limit — nothing here enforces per-process memory.
LSP_SERVER_FOOTPRINT_BYTES = 1_300_000_000

# Share of host RAM the LSP fleet may occupy before the cap bites.
# A quarter leaves the agent runtime, the language toolchains, and
# whatever the operator is actually running the other three quarters.
LSP_MEMORY_BUDGET_FRACTION = 0.25

# Floor and ceiling on the derived cap.  The floor keeps a small host
# usable at all (one server is the difference between "LSP works" and
# "LSP is off"); the ceiling stops a very large host from deriving a
# number so high it is not a bound in any practical sense.
MIN_DERIVED_MAX_SERVERS = 1
MAX_DERIVED_MAX_SERVERS = 32


def _host_memory_bytes() -> Optional[int]:
    """Total physical RAM in bytes, or ``None`` when undiscoverable.

    ``None`` is a real answer rather than a failure — the caller then
    falls back to a conservative fixed cap instead of pretending to
    have sized against a host it could not measure.
    """
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    # macOS exposes SC_PHYS_PAGES inconsistently across Python builds;
    # ``sysctl hw.memsize`` is the portable second source.
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def default_max_servers() -> int:
    """Derive the concurrent-server cap from host memory.

    A 16 GiB Mac Mini and a 128 GiB workstation must not get the same
    number — that is the point of deriving rather than hardcoding.
    16 GiB yields 3; 128 GiB yields 26.
    """
    total = _host_memory_bytes()
    if not total:
        # Undiscoverable host memory: assume small rather than large.
        # Guessing high is what produced this defect in the first place.
        return 4
    derived = int((total * LSP_MEMORY_BUDGET_FRACTION) // LSP_SERVER_FOOTPRINT_BYTES)
    return max(MIN_DERIVED_MAX_SERVERS, min(derived, MAX_DERIVED_MAX_SERVERS))


class _BackgroundLoop:
    """A daemon thread that owns one asyncio event loop.

    Provides :meth:`run` for synchronous callers — submits a coroutine
    to the loop and blocks until it finishes (or a timeout fires).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hermes-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


class LSPService:
    """The process-wide LSP service.

    Created once via :meth:`create_from_config`; the
    :func:`agent.lsp.get_service` accessor manages the singleton.
    Most callers should use that accessor rather than constructing
    :class:`LSPService` directly.
    """

    # ------------------------------------------------------------------
    # construction + factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: Optional[Dict[str, List[str]]] = None,
        env_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        init_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disabled_servers: Optional[List[str]] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        max_servers: Optional[int] = None,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        # ``0``/negative disables the respective bound.  Both are read
        # on every spawn — see ``_enforce_bounds``.  Before SCA-4389
        # ``_idle_timeout`` was stored here and never read again, so
        # the module docstring's "reaped" claim was false.
        self._idle_timeout = idle_timeout
        self._max_servers = (
            default_max_servers() if max_servers is None else int(max_servers)
        )

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        # Requests currently using a client, keyed the same way.  A key
        # with a non-zero count is never evicted, so eviction cannot
        # tear a server down mid-request.
        self._inflight: Dict[Tuple[str, str], int] = {}
        self._state_lock = threading.Lock()

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        # Both bounds are operator-overridable; absent config derives
        # the cap from host memory rather than assuming this host.
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        max_servers_cfg = lsp_cfg.get("max_servers")
        try:
            max_servers = None if max_servers_cfg is None else int(max_servers_cfg)
        except (TypeError, ValueError):
            max_servers = None
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        servers_cfg = lsp_cfg.get("servers") or {}
        disabled = []
        binary_overrides: Dict[str, List[str]] = {}
        env_overrides: Dict[str, Dict[str, str]] = {}
        init_overrides: Dict[str, Dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                cmd = sub.get("command")
                if isinstance(cmd, list) and cmd:
                    binary_overrides[name] = cmd
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {k: str(v) for k, v in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            idle_timeout=idle_timeout,
            max_servers=max_servers,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (file or cwd inside a git worktree),
        on whether any registered server matches the extension, and
        on whether the (server_id, workspace_root) pair is in the
        broken-set from a previous spawn failure.

        Files in already-broken pairs return False so the file_operations
        layer skips the LSP path entirely — no spawn attempts, no
        timeout cost — until the service is restarted (``hermes lsp
        restart``) or the process exits.
        """
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _get_or_spawn would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if (srv.server_id, per_server_root) in self._broken:
            return False
        return True

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer timeouts (e.g. server hangs during initialize) mark the
        (server_id, workspace_root) pair as broken so subsequent edits
        skip it instantly instead of re-paying the timeout cost.
        """
        if not self.enabled_for(file_path):
            return
        try:
            # Outer join budget must exceed the inner wait budget or a
            # slow-but-alive server gets falsely marked broken.
            t = max(8.0, self._wait_timeout + 3.0)
            diags = self._loop.run(self._snapshot_async(file_path), timeout=t)
            self._delta_baseline[os.path.abspath(file_path)] = diags or []
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            self._delta_baseline[os.path.abspath(file_path)] = []

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default), the result is filtered against
        any baseline previously captured via :meth:`snapshot_baseline`.
        Diagnostics present in the baseline are removed so the caller
        only sees errors introduced by the current edit.

        When ``line_shift`` is provided, baseline diagnostics are
        remapped through it before the set-difference.  This handles
        the case where the edit deleted or inserted lines, causing
        pre-existing diagnostics below the edit point to surface at
        different line numbers in the post-edit snapshot — without
        the shift, they'd all look "introduced by this edit".  Pass
        a callable built by
        :func:`agent.lsp.range_shift.build_line_shift` (pre_text,
        post_text).  Omit when pre/post content isn't available;
        the unshifted comparison still catches diagnostics that
        didn't move.

        Returns an empty list when LSP is disabled, when no workspace
        can be detected, when no server matches, or when the server
        can't be spawned.  Never raises.
        """
        if not self.enabled_for(file_path):
            return []

        # Resolve server_id eagerly so we can emit structured logs even
        # when the request errors out below.
        srv = find_server_for_file(file_path)
        server_id = srv.server_id if srv else "?"

        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t)
        except asyncio.TimeoutError as e:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []

        if diags is None:
            # The server is alive but never produced diagnostics for the
            # post-edit content within the wait budget (common for
            # tsserver on large projects).  Report "no data" rather than
            # whatever stale state is in the stores — surfacing the
            # previous edit's errors as if they were current is the
            # ghost-diagnostics bug.  The server is NOT marked broken:
            # slow is not dead, and the next edit may well succeed.
            eventlog.log_timeout(server_id, file_path, kind="fresh diagnostics")
            return []

        abs_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(abs_path) or []
            if baseline:
                if line_shift is not None:
                    # Remap baseline diagnostics into post-edit
                    # coordinates so shifted-but-otherwise-identical
                    # entries hash equal under _diag_key.  Entries
                    # that mapped into a deleted region drop out
                    # silently — they no longer apply.
                    from agent.lsp.range_shift import shift_baseline
                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(d) for d in baseline}
                diags = [d for d in diags if _diag_key(d) not in seen]
            # Roll baseline forward — next call returns deltas relative
            # to the just-emitted state, mirroring claude-code's
            # diagnosticTracking.
            try:
                fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[abs_path] = fresh

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_get_or_spawn`` task was still
        holding open.  Without this, every subsequent write would re-enter
        the spawn path and re-pay the full ``snapshot_baseline``
        timeout (8s) until the binary is fixed.

        Also kills any orphan client process that survived the cancelled
        future, and emits a single eventlog WARNING so the user knows
        which server gave up.

        ``exc`` is whatever exception the outer wrapper caught — used
        only for logging, never re-raised.
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            return
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        key = (srv.server_id, per_server_root)
        already_broken = key in self._broken
        self._broken.add(key)

        # Kill any client we managed to spawn before the timeout.  The
        # cancelled future never reached the broken-set add inside
        # ``_get_or_spawn`` so the client may still be hanging in
        # ``_clients`` with a half-initialized state.
        with self._state_lock:
            client = self._clients.pop(key, None)
            self._last_used.pop(key, None)
            self._inflight.pop(key, None)
        if client is not None:
            try:
                # Fire-and-forget shutdown — give it a second to cleanup,
                # but don't block.  We're already on a slow path.
                self._loop.run(client.shutdown(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

    def shutdown(self) -> None:
        """Tear down all clients and stop the background loop."""
        if not self._enabled:
            return
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        key = (client.server_id, client.workspace_root)
        self._acquire(key)
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            fresh = await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", e)
            return []
        finally:
            self._release(key)
        if not fresh:
            # No fresh data for the pre-edit content — an empty baseline
            # is safe: worst case the delta filter removes less, never
            # more.  Never seed the baseline from stale stores.
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _open_and_wait_async(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Open + wait for FRESH diagnostics.

        Returns the fresh diagnostic list, or ``None`` when the server
        never produced post-change data within the wait budget.  The
        distinction matters: ``[]`` means "server checked the new
        content, it's clean", ``None`` means "no verdict" — the caller
        must not substitute stale data for either.
        """
        client = await self._get_or_spawn(file_path)
        if client is None:
            return None
        key = (client.server_id, client.workspace_root)
        self._acquire(key)
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.save_file(file_path)
            fresh = await client.wait_for_diagnostics(
                file_path, version, mode=self._wait_mode, timeout=self._wait_timeout
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("open/wait failed for %s: %s", file_path, e)
            return None
        finally:
            self._release(key)
        if not fresh:
            return None
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            client = self._clients.get((srv.server_id, ws))
        if client is None:
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        srv = find_server_for_file(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        per_server_root = srv.resolve_root(file_path, ws_root)
        if per_server_root is None:
            eventlog.log_disabled(
                srv.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None  # exclude marker hit, server gated off

        key = (srv.server_id, per_server_root)
        if key in self._broken:
            return None
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                # Refresh the LRU stamp on reuse.  Before SCA-4389 only
                # spawn and post-request stamped it, so a root served
                # entirely from cache looked progressively more idle.
                self._last_used[key] = time.time()
                reuse = client
            else:
                reuse = None
            spawning = self._spawning.get(key)
        if reuse is not None:
            eventlog.log_active(srv.server_id, per_server_root)
            await self._enforce_bounds(protect=key)
            return reuse
        if spawning is not None:
            try:
                return await spawning
            except Exception:  # noqa: BLE001
                return None

        # Make room BEFORE spawning, so the fleet never transiently
        # exceeds the cap.  ``protect=key`` reserves this root's slot.
        await self._enforce_bounds(protect=key)

        # Begin spawn
        loop = asyncio.get_running_loop()
        spawn_future: asyncio.Future = loop.create_future()
        with self._state_lock:
            self._spawning[key] = spawn_future
        try:
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            )
            try:
                await client.start()
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
            self._last_used[key] = time.time()
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    # ------------------------------------------------------------------
    # eviction (SCA-4389)
    # ------------------------------------------------------------------

    def _acquire(self, key: Tuple[str, str]) -> None:
        """Mark *key* as serving a request, so eviction skips it."""
        with self._state_lock:
            self._inflight[key] = self._inflight.get(key, 0) + 1

    def _release(self, key: Tuple[str, str]) -> None:
        """Drop one in-flight reference and refresh the LRU stamp."""
        with self._state_lock:
            remaining = self._inflight.get(key, 0) - 1
            if remaining > 0:
                self._inflight[key] = remaining
            else:
                self._inflight.pop(key, None)
            self._last_used[key] = time.time()

    def _eviction_candidates(
        self, *, protect: Optional[Tuple[str, str]] = None, now: Optional[float] = None
    ) -> List[Tuple[Tuple[str, str], str]]:
        """Decide what to evict.  Pure over the service's state.

        Returns ``(key, reason)`` pairs.  Split out from the shutdown
        so the policy is testable without spawning a real server.

        *protect* is the key the current request is about to use — it
        is never a candidate even if it is the LRU, otherwise a cap of
        1 would evict the very client the caller just asked for.
        """
        now = time.time() if now is None else now
        with self._state_lock:
            keys = list(self._clients.keys())
            last_used = dict(self._last_used)
            inflight = dict(self._inflight)

        def evictable(k: Tuple[str, str]) -> bool:
            # AC6: a server draining a request is never a candidate.
            return k != protect and inflight.get(k, 0) <= 0

        victims: List[Tuple[Tuple[str, str], str]] = []
        chosen = set()

        # 1. Idle timeout — independent of population size.
        if self._idle_timeout and self._idle_timeout > 0:
            for k in keys:
                if not evictable(k):
                    continue
                age = now - last_used.get(k, now)
                if age >= self._idle_timeout:
                    victims.append((k, f"idle {int(age)}s >= {int(self._idle_timeout)}s"))
                    chosen.add(k)

        # 2. LRU cap — independent of idleness, so N simultaneously
        #    active roots still cannot exceed the host's memory.
        if self._max_servers and self._max_servers > 0:
            survivors = [k for k in keys if k not in chosen]
            # ``protect`` occupies a slot: it is either already in
            # ``_clients`` or is about to be inserted by the caller.
            projected = len(survivors) + (
                1 if protect is not None and protect not in keys else 0
            )
            if projected > self._max_servers:
                # Oldest first; unstamped keys sort oldest (0.0).
                ranked = sorted(
                    (k for k in survivors if evictable(k)),
                    key=lambda k: last_used.get(k, 0.0),
                )
                for k in ranked:
                    if projected <= self._max_servers:
                        break
                    victims.append((k, f"lru cap {self._max_servers}"))
                    projected -= 1

        return victims

    async def _enforce_bounds(self, *, protect: Optional[Tuple[str, str]] = None) -> None:
        """Evict idle and over-cap clients.  Never raises.

        Called on every spawn/reuse — the request path is the only
        clock this service has, and tying eviction to it means a cache
        that is never consulted also spawns nothing new, so its
        population cannot grow.
        """
        victims = self._eviction_candidates(protect=protect)
        if not victims:
            return
        pending = []
        for key, reason in victims:
            with self._state_lock:
                # Re-check under the lock: a request may have arrived
                # for this key between the decision and the shutdown.
                if self._inflight.get(key, 0) > 0:
                    continue
                client = self._clients.pop(key, None)
                self._last_used.pop(key, None)
            if client is None:
                continue
            eventlog.log_evicted(key[0], key[1], reason)
            pending.append(client.shutdown())
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _shutdown_async(self) -> None:
        with self._state_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
            self._inflight.clear()
        await asyncio.gather(
            *(c.shutdown() for c in clients),
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command."""
        now = time.time()
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                    # Surfacing idleness makes "is the cache evicting?"
                    # answerable from ``hermes lsp status`` rather than
                    # only from a log grep.
                    "idle_seconds": int(now - self._last_used.get(k, now)),
                    "inflight": self._inflight.get(k, 0),
                }
                for k, c in self._clients.items()
            ]
            broken = list(self._broken)
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "idle_timeout": self._idle_timeout,
            "max_servers": self._max_servers,
            "clients": clients,
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
        }


def _diag_key(d: Dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering.

    Includes the diagnostic's position range — when used together
    with :func:`agent.lsp.range_shift.shift_baseline`, the baseline
    is line-shifted into post-edit coordinates BEFORE this key is
    computed, so identical-but-shifted diagnostics hash equal.  Two
    genuinely distinct diagnostics at different lines (e.g. the same
    error class introduced at a second site) hash differently and
    are surfaced as new.

    Mirrors :func:`agent.lsp.client._diagnostic_key`; intentionally
    identical so the two layers agree on diagnostic identity.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
