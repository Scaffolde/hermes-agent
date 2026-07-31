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
import math
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

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
DEFAULT_SWEEP_INTERVAL = 60.0  # seconds between idle sweeps

# Cap derivation.  A language server's cost is dominated by the TypeScript
# program it loads, not by anything we control, so the population has to be
# bounded against the host rather than guessed.  1.3 GiB is the median
# resident footprint measured across 13 live ``typescript-language-server``
# processes on pai-mac-mini (range 1.2-1.65 GiB, each with its own tsserver
# child).  A quarter of RAM is the share we are willing to let editor
# tooling hold before it competes with the actual workload — on a 16 GiB
# host that is 4 GiB, i.e. 3 servers, against the 13 that were live when
# the box went into swap and took the self-hosted CI runner down.
LSP_CLIENT_FOOTPRINT_BYTES = 1300 * 1024 * 1024
LSP_MEMORY_BUDGET_FRACTION = 0.25
MIN_CLIENT_CAP = 1
MAX_CLIENT_CAP = 24
# Used only when host memory can't be read at all.  Deliberately the
# small-host answer: under-caching costs a few seconds of respawn, while
# over-caching costs gigabytes the host may not have.
FALLBACK_CLIENT_CAP = 3


CGROUP_MEMORY_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",  # v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
)


def _cgroup_memory_limit_bytes(paths: Tuple[str, ...] = CGROUP_MEMORY_LIMIT_PATHS) -> Optional[int]:
    """The cgroup memory ceiling in bytes, or ``None`` if unlimited/absent.

    Checked before trusting sysconf because in a memory-limited container
    ``SC_PHYS_PAGES`` reports the *node's* RAM, not the share this process
    may actually use.  cgroup v2 writes ``max`` for "no limit"; v1 writes a
    sentinel near 2**63, so implausibly large values are treated as absent.

    *paths* is injectable so the sentinel handling is testable off-Linux.
    """
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except (OSError, ValueError):
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        # v1's "unlimited" sentinel is page-aligned LONG_MAX; anything at
        # petabyte scale is that sentinel rather than a real allocation.
        if limit <= 0 or limit >= (1 << 50):
            continue
        return limit
    return None


def host_memory_bytes() -> Optional[int]:
    """Memory this process may actually use, or ``None`` if undeterminable.

    ``SC_PHYS_PAGES``/``SC_PAGE_SIZE`` are present on Linux and macOS.
    Anything else (Windows, an exotic libc, a sandbox that stubs sysconf)
    falls through to ``None`` and the caller uses a conservative default.

    A cgroup limit, when present, wins over the sysconf total whenever it
    is smaller: deriving the cap from 64 GiB of node RAM inside a 4 GiB
    container permits a client population that OOMs the container.
    """
    total: Optional[int]
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        total = None
    else:
        total = pages * page_size if pages > 0 and page_size > 0 else None

    limit = _cgroup_memory_limit_bytes()
    if limit is not None:
        return limit if total is None else min(total, limit)
    return total


def default_max_clients(total_bytes: Optional[int] = None) -> int:
    """How many concurrent language servers this host can afford.

    Pass ``total_bytes`` to compute the cap for a hypothetical host; omit
    it to measure the current one.  The result is always clamped to
    ``[MIN_CLIENT_CAP, MAX_CLIENT_CAP]`` — one server is the floor because
    a cap of zero would disable the feature outright, and the ceiling
    keeps a very large host from accumulating an unbounded pool simply
    because it has the headroom to hide the growth.
    """
    if total_bytes is None:
        total_bytes = host_memory_bytes()
    if not total_bytes or total_bytes <= 0:
        return FALLBACK_CLIENT_CAP
    budget = int(total_bytes * LSP_MEMORY_BUDGET_FRACTION)
    derived = budget // LSP_CLIENT_FOOTPRINT_BYTES
    return max(MIN_CLIENT_CAP, min(MAX_CLIENT_CAP, int(derived)))


def _coerce_positive(value: Any, default: float) -> float:
    """Parse a config bound, falling back to *default* on anything unusable.

    Garbage must not silently disable a bound — that is how the
    pre-eviction behaviour comes back without anyone noticing.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    # ``.nan``/``.inf`` are valid YAML and survive ``float()``, but both
    # slip past the ``<= 0`` test and then blow up in the ``int()`` at the
    # call site (ValueError / OverflowError).  That exception escapes
    # ``create_from_config`` and takes the whole LSP path down, so a
    # merely-odd config would break write/patch instead of falling back.
    if not math.isfinite(parsed):
        return default
    if parsed <= 0:
        return default
    return parsed


def _coerce_non_negative(value: Any, default: float) -> float:
    """Like :func:`_coerce_positive` but accepts an explicit ``0``.

    Used for the idle timeout, where ``0`` is a meaningful opt-out ("pin
    servers for the life of the process") rather than a typo.  The cap has
    no such reading — a cap of zero would disable the feature — so it uses
    the strictly-positive form and clamps instead.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < 0:
        return default
    return parsed


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

    def spawn(self, coro):
        """Schedule ``coro`` on the loop without waiting for it.

        Returns the :class:`concurrent.futures.Future` so the caller can
        cancel it later (the reaper is cancelled at shutdown), or ``None``
        if the loop isn't running.  Unlike :meth:`run` this never blocks —
        a long-lived background task submitted through ``run`` would
        deadlock the caller forever.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            return None
        return safe_schedule_threadsafe(coro, self._loop)

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
        max_clients: Optional[int] = None,
        sweep_interval: float = DEFAULT_SWEEP_INTERVAL,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = idle_timeout
        self._max_clients = (
            default_max_clients() if max_clients is None
            else max(MIN_CLIENT_CAP, min(MAX_CLIENT_CAP, int(max_clients)))
        )
        self._sweep_interval = sweep_interval

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        # Strong refs to in-flight background cap-enforcement tasks; see
        # the ``create_task`` call in ``_get_or_spawn``.
        self._cap_tasks: set = set()
        # Outstanding requests per key.  A client with a non-zero count is
        # mid-request and must never be evicted underneath it — the caller
        # is blocked on a diagnostics round-trip that would come back as a
        # spurious "server died" instead of an answer.
        self._inflight: Dict[Tuple[str, str], int] = {}
        self._state_lock = threading.Lock()
        self._reaper: Any = None

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        # The reaper is what makes the idle timeout real.  Before this
        # existed, ``_idle_timeout`` and ``_last_used`` were both written
        # and never read, so the cache grew for the life of the gateway.
        # It runs whenever the service is enabled, not only when an idle
        # timeout is set: ``idle_timeout: 0`` opts out of *idleness* but
        # not out of the cap, and the cap needs a periodic enforcer to
        # collect overage left behind by a concurrent burst.
        if self._enabled:
            self._reaper = self._loop.spawn(self._reaper_loop())

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

        # Eviction bounds.  Defaults are host-derived rather than fixed:
        # a 16 GiB Mac Mini and a 128 GiB workstation should not carry the
        # same number of language servers.  ``max_clients`` omitted (or
        # unparseable) means "measure this host".
        idle_timeout = _coerce_non_negative(
            lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT), DEFAULT_IDLE_TIMEOUT
        )
        sweep_interval = _coerce_positive(
            lsp_cfg.get("sweep_interval", DEFAULT_SWEEP_INTERVAL), DEFAULT_SWEEP_INTERVAL
        )
        raw_cap = lsp_cfg.get("max_clients")
        max_clients = (
            None if raw_cap is None
            else int(_coerce_positive(raw_cap, default_max_clients()))
        )

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
            max_clients=max_clients,
            sweep_interval=sweep_interval,
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
            # Drop the eviction bookkeeping with it — a key left in
            # ``_last_used`` with no client behind it is exactly the stale
            # state that made the idle timeout unauditable.
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
        reaper = self._reaper
        self._reaper = None
        if reaper is not None:
            reaper.cancel()
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    # ------------------------------------------------------------------
    # eviction
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        """The clock every eviction age is measured against.

        Monotonic, never wall time: an NTP correction, a VM resume, or a
        manual clock change must not be able to suspend the reaper (a
        backward step leaves every stored stamp ahead of the cutoff) or
        evict clients that were just used (a forward step).  Routing all
        eviction bookkeeping through one seam keeps the two sides from
        drifting onto different clocks.
        """
        return time.monotonic()

    def _acquire(self, key: Tuple[str, str]) -> None:
        """Mark a request as in-flight against *key*.

        Refcounted rather than a boolean: two concurrent edits in the same
        project share one client, and the first to finish must not expose
        it to eviction while the second is still waiting.
        """
        with self._state_lock:
            self._inflight[key] = self._inflight.get(key, 0) + 1

    def _release(self, key: Tuple[str, str]) -> None:
        with self._state_lock:
            remaining = self._inflight.get(key, 0) - 1
            if remaining > 0:
                self._inflight[key] = remaining
            else:
                self._inflight.pop(key, None)

    def sweep_idle_now(self) -> List[Tuple[str, str]]:
        """Run one idle sweep synchronously; return the keys evicted.

        The reaper calls the same coroutine on its timer — this is the
        hand-crank for tests and for ``hermes lsp`` tooling, not a
        substitute for the timer.
        """
        if not self._enabled:
            return []
        try:
            return self._loop.run(self._sweep_idle_async(), timeout=15.0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP idle sweep failed: %s", e)
            return []

    def enforce_cap_now(self) -> List[Tuple[str, str]]:
        """Evict least-recently-used clients until the cap is satisfied."""
        if not self._enabled:
            return []
        try:
            return self._loop.run(self._enforce_cap_async(), timeout=15.0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP cap enforcement failed: %s", e)
            return []

    async def _evict(self, key: Tuple[str, str], reason: str) -> bool:
        """Shut down the client at *key* and drop its bookkeeping.

        Returns False if it vanished under us (a concurrent shutdown or a
        second sweep) so callers can keep their evicted-list honest.
        """
        with self._state_lock:
            if self._inflight.get(key, 0) > 0:
                return False
            client = self._clients.pop(key, None)
            self._last_used.pop(key, None)
        if client is None:
            return False
        eventlog.log_evicted(key[0], key[1], reason)
        try:
            await client.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("evicted client %s failed to shut down cleanly: %s", key, e)
        return True

    def _evictable(self) -> List[Tuple[Tuple[str, str], float]]:
        """(key, last_used) for clients with no outstanding request,
        oldest first."""
        with self._state_lock:
            candidates = [
                (key, self._last_used.get(key, 0.0))
                for key in self._clients
                if self._inflight.get(key, 0) == 0
            ]
        candidates.sort(key=lambda kv: kv[1])
        return candidates

    async def _sweep_idle_async(self) -> List[Tuple[str, str]]:
        if self._idle_timeout <= 0:
            return []
        cutoff = self._now() - self._idle_timeout
        evicted: List[Tuple[str, str]] = []
        for key, last_used in self._evictable():
            if last_used > cutoff:
                # Sorted oldest-first, so the first fresh one ends the scan.
                break
            if await self._evict(key, f"idle > {int(self._idle_timeout)}s"):
                evicted.append(key)
        return evicted

    async def _enforce_cap_async(
        self, protect: Optional[Tuple[str, str]] = None
    ) -> List[Tuple[str, str]]:
        """Drain to ``_max_clients``, evicting least-recently-used first.

        *protect* is the key that just spawned: evicting the client the
        caller is about to use would turn a cap into an infinite
        spawn/evict loop.
        """
        evicted: List[Tuple[str, str]] = []
        while True:
            with self._state_lock:
                overage = len(self._clients) - self._max_clients
            if overage <= 0:
                break
            victim = next(
                (key for key, _ in self._evictable() if key != protect),
                None,
            )
            if victim is None:
                # Everything left is in-flight or protected.  Going over
                # the cap briefly is the right trade against killing a
                # live request; the next sweep collects the slack.
                logger.debug(
                    "LSP cap %d exceeded by %d but all clients are busy",
                    self._max_clients,
                    overage,
                )
                break
            if await self._evict(victim, f"lru cap {self._max_clients}"):
                evicted.append(victim)
            else:
                break
        return evicted

    async def _reaper_loop(self) -> None:
        """Periodically evict idle and over-cap clients, for the service's life."""
        interval = max(1.0, min(self._sweep_interval, max(self._idle_timeout, 1.0)))
        while True:
            try:
                await asyncio.sleep(interval)
                await self._sweep_idle_async()
                # The cap must be re-checked every iteration, not only at
                # spawn.  When requests against more than ``_max_clients``
                # roots overlap, every existing client is in-flight and the
                # spawn-time enforcement finds no evictable victim, so the
                # overage outlives the burst.  Sweeping alone cannot collect
                # it: those clients were just touched, and with
                # ``idle_timeout: 0`` the sweep is a no-op forever.
                await self._enforce_cap_async()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # A reaper that dies on one bad client silently restores the
                # unbounded-growth behaviour, so it never exits on error.
                logger.debug("LSP reaper iteration failed: %s", e)

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        # ``_get_or_spawn`` hands back a client with the in-flight refcount
        # already held; releasing it is this caller's responsibility.
        key = (client.server_id, client.workspace_root)
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            fresh = await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", e)
            return []
        finally:
            # Touch before releasing: a slow or failed request is still
            # activity on this root, and counting it as idle would make a
            # struggling server the first thing the reaper kills.
            with self._state_lock:
                if key in self._clients:
                    self._last_used[key] = self._now()
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
            with self._state_lock:
                if key in self._clients:
                    self._last_used[key] = self._now()
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
                # Acquire under the same lock that read the client, so an
                # eviction can never slip between the lookup and the
                # refcount and shut the server down under this caller.
                self._inflight[key] = self._inflight.get(key, 0) + 1
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                client = await spawning
            except Exception:  # noqa: BLE001
                return None
            if client is None:
                return None
            self._acquire(key)
            return client

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
                self._last_used[key] = self._now()
                self._inflight[key] = self._inflight.get(key, 0) + 1
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            # Bound the population the moment it grows.  ``protect=key``
            # keeps the cap from evicting the client this caller is about
            # to use, which would otherwise spin spawn/evict forever once
            # the cap is saturated.
            #
            # Deliberately not awaited: the victim's shutdown can take up
            # to its own shutdown timeout, and charging that to the caller
            # that just spawned pushes it past the outer diagnostic budget.
            # The wrappers then time out, mark the root broken, and kill a
            # client that started perfectly well — disabling that root for
            # the rest of the service's life.  Eviction is background work;
            # the reaper re-checks the cap regardless.
            #
            # ``create_task`` (not ``_loop.spawn``) because we are already
            # on the loop here, and a strong reference is retained because
            # the event loop only holds a weak one — an unreferenced task
            # can be collected mid-shutdown, silently skipping eviction.
            task = asyncio.create_task(self._enforce_cap_async(protect=key))
            self._cap_tasks.add(task)
            task.add_done_callback(self._cap_tasks.discard)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

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
        now = self._now()
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                    # Surfacing idleness and busyness makes the eviction
                    # bounds auditable from ``hermes lsp status`` — an
                    # operator can see why a server is or isn't a
                    # candidate without reading this file.
                    "idle_seconds": round(now - self._last_used.get(k, now), 1),
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
            "max_clients": self._max_clients,
            "sweep_interval": self._sweep_interval,
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


__all__ = [
    "LSPService",
    "DEFAULT_IDLE_TIMEOUT",
    "DEFAULT_SWEEP_INTERVAL",
    "LSP_CLIENT_FOOTPRINT_BYTES",
    "LSP_MEMORY_BUDGET_FRACTION",
    "MIN_CLIENT_CAP",
    "MAX_CLIENT_CAP",
    "FALLBACK_CLIENT_CAP",
    "host_memory_bytes",
    "default_max_clients",
]
