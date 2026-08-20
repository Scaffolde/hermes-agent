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
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent import cgroup_memory
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
MIN_IDLE_TIMEOUT = 30  # floor for config values; must exceed any per-op wait budget

# Cap derivation.  ``idle_timeout`` bounds how *long* a client survives; it
# does not bound how *many* exist at once.  Thirteen live
# ``typescript-language-server`` processes holding ~16 GiB put pai-mac-mini
# into swap while every one of them was inside its idle window, so the
# reaper could not have helped.  A language server's cost is dominated by
# the program it loads, not by anything we control, so the population has
# to be bounded against the host rather than guessed.  1.3 GiB is the
# median resident footprint measured across those 13 processes (range
# 1.2-1.65 GiB, each with its own tsserver child).  A quarter of RAM is the
# share we are willing to let editor tooling hold before it competes with
# the actual workload — on a 16 GiB host that is 4 GiB, i.e. 3 servers.
LSP_CLIENT_FOOTPRINT_BYTES = 1300 * 1024 * 1024
LSP_MEMORY_BUDGET_FRACTION = 0.25
MIN_CLIENT_CAP = 1
MAX_CLIENT_CAP = 24

_MIB = 1024 * 1024

# Per-type cost.  Charging every server the typescript figure is what made
# the derived cap 3 on a 16 GiB host while the measured healthy working set
# was 7 (SCA-4688): the real fleet is mostly cheap servers, and a yaml
# server was being charged 26x what it actually costs.
#
# Every value below comes from ``scripts/measure_lsp_footprint.py`` — a real
# spawn against a real project, physical footprint summed over the process
# subtree, peak of three samples after a 20s settle.  Raw readings are in
# ``docs/lsp-footprint-measurements.json``.  RSS is not usable here: the
# Node-based servers run behind a shim that reads 5-11 MB while the actual
# analysis lives in a child process.
#
# Each stored value is its measurement rounded *up*.  Under-charging is the
# direction that re-opens SCA-4389, so headroom always makes the cap
# stricter than the measurement, never looser.
LSP_SERVER_FOOTPRINT_BYTES: Dict[str, int] = {
    # measured 1688.4 MiB against an 8.5k-file TypeScript project — the same
    # workload class as the 13-process SCA-4389 incident, whose per-process
    # range was 1.2-1.65 GiB.  A small project measures 500 MiB, but a cap
    # sized from the cheap case is a cap that fails exactly when it matters.
    "typescript": 1700 * _MIB,
    # measured 633.7 MiB against an 8.5k-file Python project.
    "pyright": 800 * _MIB,
    # measured 41.2 MiB.  Charged 150 to leave room for a larger schema set.
    "yaml-language-server": 150 * _MIB,
    # measured 49.1 MiB.  Charged 150 for the same reason.
    "bash-language-server": 150 * _MIB,
}


class _DeriveBudget:
    """Sentinel: derive the byte budget from host memory.

    Distinct from ``None``, which explicitly *disables* the byte half of
    the bound.  A plain ``None`` default could not tell "read the host"
    apart from "this fleet has no byte budget".
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<derive-from-host>"


_DERIVE_BUDGET = _DeriveBudget()


def footprint_for(server_id: str) -> int:
    """What one *server_id* costs against the memory budget, in bytes.

    An unmeasured type charges ``LSP_CLIENT_FOOTPRINT_BYTES`` — the value
    every type was charged before this table existed.  So an unknown server
    is admitted exactly as conservatively as it is today, and no type can
    become cheaper merely by being absent from the table.  Types known to be
    heavy but not yet measured (rust-analyzer, gopls, jdtls) are therefore
    left unchanged rather than optimistically discounted.
    """
    return LSP_SERVER_FOOTPRINT_BYTES.get(server_id, LSP_CLIENT_FOOTPRINT_BYTES)


def memory_budget_bytes(total_bytes: Optional[int] = None) -> Optional[int]:
    """The byte budget the whole fleet is admitted against.

    Deliberately the *same* share of host RAM as before per-type accounting
    existed: the point is to fit more servers into the existing budget
    because most of them are cheap, never to raise the ceiling.
    """
    if total_bytes is None:
        total_bytes = host_memory_bytes()
    if not total_bytes or total_bytes <= 0:
        return None
    return int(total_bytes * LSP_MEMORY_BUDGET_FRACTION)
# Used only when host memory can't be read at all.  Deliberately the
# small-host answer: under-caching costs a few seconds of respawn, while
# over-caching costs gigabytes the host may not have.
FALLBACK_CLIENT_CAP = 3

# How long a spawning request waits for a victim's shutdown before
# leaving it to drain in the background.  Cap enforcement runs inside
# ``get_diagnostics_sync``'s outer budget, which allows the diagnostics
# wait ``wait_timeout`` plus a 2s guard band; a victim that spends the
# whole guard band leaves the wait unable to use the budget it was
# promised, and the caller is told the server had nothing to say.  A
# healthy shutdown is one ``shutdown`` round-trip — milliseconds — while
# a wedged one costs the client's 2s request timeout plus
# ``SHUTDOWN_GRACE``.  Half a second sits clear of both: normal
# evictions still complete before the new server starts, so the fleet
# does not overlap, and a wedged one is bounded well inside the guard
# band instead of consuming it.
EVICTION_HANDOFF_BUDGET = 0.5


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

    limit = cgroup_memory.cgroup_memory_limit_bytes()
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

    This is now the *count* half of a two-part bound.  The memory half
    (:meth:`LSPService._memory_overage_locked`) charges each live server
    its own measured footprint, and that is what actually holds the fleet
    inside the host's RAM.  So this divides by the cheapest server rather
    than the most expensive one: sizing the count ceiling off typescript
    made it bind at 3 on a 16 GiB host and evict during ordinary
    multi-worktree work, even when every live server was a 41 MiB yaml
    process that fitted the budget many times over (SCA-4688).
    """
    if total_bytes is None:
        total_bytes = host_memory_bytes()
    if not total_bytes or total_bytes <= 0:
        return FALLBACK_CLIENT_CAP
    budget = int(total_bytes * LSP_MEMORY_BUDGET_FRACTION)
    cheapest = min([LSP_CLIENT_FOOTPRINT_BYTES, *LSP_SERVER_FOOTPRINT_BYTES.values()])
    derived = budget // cheapest
    return max(MIN_CLIENT_CAP, min(MAX_CLIENT_CAP, int(derived)))


def resolve_max_clients(raw: Any) -> int:
    """Turn a configured ``lsp.max_clients`` into an enforceable cap.

    ``None`` means "derive from the host".  Anything unusable — zero, a
    negative, a string, a ``bool``, ``.nan``/``.inf`` (all valid YAML that
    survive ``float()``) — also derives, because garbage must not silently
    restore the unbounded accumulation this cap exists to stop.
    """
    if raw is None or isinstance(raw, bool):
        return default_max_clients()
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return default_max_clients()
    if not math.isfinite(parsed) or parsed <= 0:
        return default_max_clients()
    return max(MIN_CLIENT_CAP, min(MAX_CLIENT_CAP, int(parsed)))


def resolve_idle_timeout(raw: Any) -> float:
    """Turn a configured ``lsp.idle_timeout`` into an enforceable deadline.

    The sibling of :func:`resolve_max_clients`, and deliberately shaped
    like it: the two knobs bound the same fleet on different axes
    (``max_clients`` how many, this one how long), and coercing them in
    two different places is how they drifted apart in the first place.

    ``.nan`` and ``.inf`` are the cases that motivated this.  Both are
    valid YAML, both survive ``float()`` without raising, and both slip
    a ``0 < x < MIN_IDLE_TIMEOUT`` clamp.  ``nan`` then fails the
    ``idle_timeout > 0`` start guard so the reaper never starts, and
    ``inf`` starts a reaper whose cutoff is ``-inf``, which no
    ``_last_used`` stamp is ever below — so it sweeps forever and reaps
    nothing.  Either way the fleet pins at ``max_clients`` and never
    drains, silently.

    ``0`` is the one non-positive value kept as written: the user guide
    documents it as "hold every server's index warm for the life of the
    process".  A negative reaches that same disabled state by accident
    rather than by request, so it derives the default instead.

    A ``bool`` is rejected before coercion for the same reason, and it is
    the sharpest case: YAML 1.1 resolves ``off``/``no`` to ``False``, and
    ``bool`` subclasses ``int``, so ``idle_timeout: off`` would sail
    through ``float()`` as ``0.0`` and read as the documented disable —
    silently reaching the dead reaper above by a different door.  ``on``
    would land at ``1.0`` and clamp up to ``MIN_IDLE_TIMEOUT``.  Neither
    is a duration anyone can have meant.
    """
    if raw is None or isinstance(raw, bool):
        return float(DEFAULT_IDLE_TIMEOUT)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_IDLE_TIMEOUT)
    if not math.isfinite(parsed) or parsed < 0:
        return float(DEFAULT_IDLE_TIMEOUT)
    if parsed == 0:
        return 0.0
    # Below the per-operation wait budget a sweep could reap a client
    # mid-flight; the resulting outer timeout marks the (server,
    # workspace) pair broken for the process lifetime.
    return max(float(MIN_IDLE_TIMEOUT), parsed)


def _bound_was_overridden(raw: Any, resolved: float) -> bool:
    """True when a configured bound did not survive resolution as written.

    Compares numerically so ``"4"`` and ``4`` do not read as a change —
    otherwise every string-typed config would warn about an override
    that never happened, and a warning on every startup is a warning
    nobody reads.

    A ``bool`` is always an override, decided before the numeric
    comparison can excuse it.  ``MIN_CLIENT_CAP`` is 1, so on a small
    host ``max_clients: on`` resolves to a default of 1 and would
    compare equal to ``float(True)`` — leaving the one case where a
    rejected value is corrected in silence.
    """
    if raw is None:
        return False
    if isinstance(raw, bool):
        return True
    try:
        return float(raw) != float(resolved)
    except (TypeError, ValueError):
        return True


def _log_cap_enforcement_result(fut: Future) -> None:
    """Drain a fire-and-forget cap sweep so its failure is not silent.

    Nobody calls ``.result()`` on a detached enforcement future, so an
    exception inside the sweep would otherwise vanish — and a silently
    dead sweep is exactly the failure mode this whole path exists to
    close.  Never re-raises: this runs on the loop thread.
    """
    try:
        fut.result()
    except Exception as e:  # noqa: BLE001
        logger.debug("detached LSP cap enforcement failed: %s", e)


def _idle_clock() -> float:
    """Clock for idle bookkeeping — suspend-inclusive, never the wall clock.

    ``_last_used`` and the reaper's cutoff are only ever compared to each
    other, so they need elapsed time, not a date.  The wall clock supplies
    neither guarantee: NTP correction and sleep/wake both step it, and a
    backwards step of more than ``idle_timeout`` pushes every cutoff into
    the past and stalls reaping until the clock catches up (a forwards step
    does the opposite and reaps servers that are still in use).  Both
    failure modes are silent, and reviving unbounded accumulation is the
    exact leak the reaper exists to close.

    ``CLOCK_MONOTONIC`` fixes the stepping but introduces the mirror-image
    bug: on Linux it *stops* while the machine is suspended.  A laptop that
    sleeps for longer than ``idle_timeout`` therefore wakes with every
    ``_last_used`` stamp still inside the window, and each sleep/wake cycle
    leaks another generation of servers — worse than the wall clock, which
    at least aged them.  ``CLOCK_BOOTTIME`` is monotonic *and* counts
    suspended time, so it is the only source that satisfies both halves.

    Platforms without ``CLOCK_BOOTTIME`` (macOS, some BSDs) fall back to
    ``monotonic()``.  The resolution is deliberately per-call rather than
    cached at import: it keeps the module's ``time`` reference patchable by
    the suspend/NTP tests, and a ``getattr`` is noise next to a sweep.
    """
    boottime_id = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime_id is not None:
        try:
            return time.clock_gettime(boottime_id)
        except (OSError, ValueError, AttributeError):
            # Kernel or libc refused the clock — degrade rather than take
            # the whole reaper down with an unhandled error.
            pass
    return time.monotonic()


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

    def schedule(self, coro) -> Optional[Future]:
        """Submit a coroutine to the loop and return immediately.

        Fire-and-forget counterpart to :meth:`run`: the caller is not
        charged for the coroutine's runtime.  Returns the
        :class:`concurrent.futures.Future` so callers can attach a
        done-callback, or ``None`` when the loop is not running (the
        coroutine is closed rather than leaked).
        """
        from agent.async_utils import safe_schedule_threadsafe
        return safe_schedule_threadsafe(
            coro,
            self._loop,
            logger=logger,
            log_message="Failed to schedule LSP coroutine",
        )

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
        memory_budget: Any = _DERIVE_BUDGET,
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
        self._max_clients = resolve_max_clients(max_clients)
        # The byte half of the bound.  ``None`` on a host whose memory
        # cannot be read, where the count cap is the only bound available.
        #
        # Injectable because deriving it from host RAM makes every test
        # that exercises the *count* cap host-dependent: on a 1 GiB CI
        # worker the derived budget is 256 MiB, so a fleet of fake
        # pyright clients is charged 800 MiB each and the byte bound
        # evicts down to the floor before the count cap under test can
        # bind.  ``memory_budget=None`` disables the byte half outright;
        # an explicit int pins it.
        self._memory_budget = (
            memory_budget_bytes()
            if memory_budget is _DERIVE_BUDGET
            else memory_budget
        )

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        # Shutdowns whose client has left ``_clients`` but whose process
        # has not exited yet.  The mirror of ``_spawning``: it reserves
        # the slot of a server that is about to stop, so a detached
        # shutdown never reads as a free slot the host has not reclaimed.
        #
        # Keyed by TASK, not by ``(server_id, workspace_root)``.  Every
        # reader of this map counts *processes* — ``_overage_locked``
        # adds ``len()`` to the resident total, and ``_shutdown_async``
        # awaits its members — and one workspace can have several
        # shutdowns resident at once: evict a wedged client, respawn the
        # same root, evict the replacement before the first drain lands.
        # A per-key dict collapses those into one entry, so the surviving
        # process is neither counted against the cap nor awaited at
        # service stop.  Task identity is what is one-per-process.
        self._shutting_down: Dict[asyncio.Task, Tuple[str, str]] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        # Refcounted per key rather than a boolean: two concurrent edits in
        # the same project share one client, and the first to finish must
        # not expose it to eviction while the second is still waiting.
        self._inflight: Dict[Tuple[str, str], int] = {}
        self._state_lock = threading.Lock()
        self._idle_reaper_task: Optional[asyncio.Task] = None

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        if self._enabled and self._idle_timeout > 0:
            self._loop.run(self._start_idle_reaper(), timeout=2.0)

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
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
        raw_idle_timeout = lsp_cfg.get("idle_timeout")
        idle_timeout = resolve_idle_timeout(raw_idle_timeout)
        # Omitted (or unusable) derives the cap from host memory rather
        # than pinning a constant: a 64 GiB CI box and a 16 GiB laptop
        # cannot afford the same number of language servers.
        raw_max_clients = lsp_cfg.get("max_clients")
        max_clients = resolve_max_clients(raw_max_clients)
        # WARNING, not debug: an overridden bound is the operator
        # believing a limit they configured is in force when a different
        # one is.  That belief is what makes a pinned fleet or an
        # unreaped one undiagnosable.
        if _bound_was_overridden(raw_idle_timeout, idle_timeout):
            logger.warning(
                "lsp.idle_timeout=%r is not usable; using %.1fs instead",
                raw_idle_timeout,
                idle_timeout,
            )
        if _bound_was_overridden(raw_max_clients, max_clients):
            logger.warning(
                "lsp.max_clients=%r is not usable; using %d instead",
                raw_max_clients,
                max_clients,
            )
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
            max_clients=max_clients,
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
        self._touch(client)
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
        self._touch(client)
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
                self._last_used[key] = _idle_clock()
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await spawning
            except Exception:  # noqa: BLE001
                return None

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
            # Make room BEFORE starting the process, not after.  ``key``
            # is already registered in ``_spawning``, so the cap counts
            # this server as occupying its slot and drains to leave room
            # for it.  Evicting afterwards would let the fleet hold
            # cap + 1 live servers, and the peak is what this bounds.
            # No ``protect`` here: nothing is being returned yet, and a
            # stale dead client under ``key`` should be reclaimed.
            # Bounded handoff: this runs inside the caller's diagnostics
            # budget, so a victim that will not die must not be able to
            # spend it (SCA-4628).
            await self._enforce_cap_async(handoff=EVICTION_HANDOFF_BUDGET)
            try:
                await client.start()
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
                self._last_used[key] = _idle_clock()
            # Second sweep, protecting the client this call returns:
            # concurrent spawns for other roots can still have raced the
            # reservation above. Evicting the caller's own client would
            # make every spawn immediately undo itself.
            await self._enforce_cap_async(
                protect=key, handoff=EVICTION_HANDOFF_BUDGET
            )
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    async def _start_idle_reaper(self) -> None:
        self._idle_reaper_task = asyncio.create_task(self._idle_reaper_loop())

    # ------------------------------------------------------------------
    # eviction: in-flight accounting and the concurrent cap
    # ------------------------------------------------------------------

    def _acquire(self, key: Tuple[str, str]) -> None:
        """Mark a request as in-flight against *key*."""
        with self._state_lock:
            self._inflight[key] = self._inflight.get(key, 0) + 1

    def _release(self, key: Tuple[str, str]) -> None:
        with self._state_lock:
            remaining = self._inflight.get(key, 0) - 1
            if remaining > 0:
                self._inflight[key] = remaining
                return
            self._inflight.pop(key, None)
            # This release may have produced the first evictable client
            # since ``_enforce_cap_async`` broke out over the cap with
            # everything busy.  Nothing else closes that escape hatch:
            # the spawn path only runs on the next spawn, and the idle
            # reaper is disabled outright at ``idle_timeout: 0`` — a
            # supported setting for keeping indexes warm.  Without this,
            # "briefly over the cap" becomes "over the cap forever",
            # which is the population blow-up the cap exists to bound.
            over_cap = self._overage_locked() > 0
        if not over_cap:
            return
        self._schedule_cap_enforcement()

    def _overage_locked(self, *, exclude_draining: bool = False) -> int:
        """How far the fleet is over the cap.  Caller holds ``_state_lock``.

        Counts keys reserved in ``_spawning`` but not yet in
        ``_clients`` — a server that is about to exist occupies its
        slot, matching :meth:`_enforce_cap_async`.

        Entries in ``_shutting_down`` count for the mirror-image reason: an
        evicted server is out of ``_clients`` but its process is still
        resident until the shutdown lands, and the cap bounds resident
        processes, not map entries.  Without this the fleet would look
        like it had free slots the host has not actually got back.

        *exclude_draining* answers the different question the sweep's stop
        condition needs: how far over we would STILL be once every
        shutdown already in flight has landed.  See
        :meth:`_binding_overage_locked`.
        """
        return self._binding_overage_locked(exclude_draining=exclude_draining)[0]

    def _binding_overage_locked(
        self, *, exclude_draining: bool = False
    ) -> Tuple[int, str]:
        """``(overage, reason)`` — how far over, and which bound says so.

        The reason travels with the number because the two halves of the
        bound fail for different operational causes and an operator has to
        tell them apart: "cap 24 exceeded" logged against a three-client
        fleet is nonsense, and it is what a byte-triggered eviction used
        to report on a 16 GiB host, where three typescript servers blow
        the 4 GiB budget while sitting far below the derived count cap.
        """
        pending = sum(1 for k in self._spawning if k not in self._clients)
        draining = 0 if exclude_draining else len(self._shutting_down)
        count_overage = (
            len(self._clients)
            + pending
            + draining
            - self._max_clients
        )
        memory_overage = self._memory_overage_locked(
            exclude_draining=exclude_draining
        )
        if memory_overage > count_overage:
            budget_mib = (self._memory_budget or 0) // _MIB
            return memory_overage, f"memory budget {budget_mib} MiB exceeded"
        return count_overage, f"cap {self._max_clients} exceeded"

    def _memory_overage_locked(self, *, exclude_draining: bool = False) -> int:
        """How many clients must go to fit the byte budget.  Holds the lock.

        The count cap alone cannot express "7 cheap servers are fine but 3
        typescript servers are not" — that was SCA-4688.  This charges each
        live server its own measured footprint and reports how many LRU
        victims it would take to get back inside the budget, which is the
        unit :meth:`_enforce_cap_async` already loops on.

        Counts ``_spawning`` and ``_shutting_down`` for exactly the reasons
        the count overage does: a server about to exist already owes its
        memory, and an evicted one still holds its pages until the process
        actually goes.

        Never asks for the last client's eviction.  A single server larger
        than the whole budget — typescript on a small host — would otherwise
        be evicted the moment it spawned and respawned by the next request,
        forever, which costs more than the memory it was meant to save.
        ``MIN_CLIENT_CAP`` is the same floor the count path already applies.
        """
        budget = self._memory_budget
        if not budget:
            return 0
        total = sum(footprint_for(sid) for sid, _root in self._clients)
        total += sum(
            footprint_for(sid)
            for (sid, _root) in self._spawning
            if (sid, _root) not in self._clients
        )
        # ``_shutting_down`` is keyed by task; the client key is the value.
        if not exclude_draining:
            total += sum(
                footprint_for(sid) for sid, _root in self._shutting_down.values()
            )
        if total <= budget:
            return 0
        # Only live clients can be handed to the sweep: a pending spawn has
        # not reached its caller and a shutting-down one is already leaving.
        # Ordered least-recently-used first, matching ``_evictable``, so this
        # is what the sweep will actually have to remove rather than an
        # optimistic best case.
        candidates = [key for key in self._clients if key not in self._spawning]
        candidates.sort(key=lambda key: self._last_used.get(key, 0.0))
        # The floor applies to the fleet that will EXIST, which includes a
        # pending spawn — not to ``_clients`` alone.  Counting only live
        # clients let one live server plus one pending one report zero
        # overage: ``1 - MIN_CLIENT_CAP`` is nothing removable, so the
        # pre-start sweep in ``_get_or_spawn`` admitted the new process
        # and both indexed concurrently.  On a 4 GiB host that is two
        # typescript servers (3.4 GiB) against a 1 GiB budget — exactly
        # the peak this bound exists to prevent.  Evicting the live one
        # here still leaves the pending one, so the floor is honoured.
        eventual = len(self._clients) + sum(
            1 for k in self._spawning if k not in self._clients
        )
        removable = max(0, eventual - MIN_CLIENT_CAP)
        needed = 0
        for key in candidates[:removable]:
            total -= footprint_for(key[0])
            needed += 1
            if total <= budget:
                break
        return needed

    def _schedule_cap_enforcement(self) -> None:
        """Re-run cap enforcement on the background loop, fire-and-forget.

        Deliberately not awaited: the request that just finished must
        not be charged for a victim's shutdown.  Making the releasing
        caller wait would push the eviction cost into the diagnostics
        timeout budget it is trying to leave.
        """
        if not self._enabled:
            return
        fut = self._loop.schedule(self._enforce_cap_async())
        if fut is not None:
            fut.add_done_callback(_log_cap_enforcement_result)

    def _evictable(self) -> List[Tuple[Tuple[str, str], float]]:
        """(key, last_used) for clients safe to shut down, oldest first.

        Two things disqualify a client.  An outstanding request is the
        obvious one.  The other is subtler: ``_get_or_spawn`` publishes
        into ``_clients`` and only *then* returns, and its caller runs
        ``_acquire`` after that return.  Through that whole window the
        client is live with an in-flight count of zero, so it would
        otherwise look idle to a concurrent spawn's sweep — which
        carries no ``protect`` for it, since ``protect`` is per-call and
        names only the sweeping caller's own key.  Evicting there hands
        the victim's caller an already-shut-down client and silently
        drops its diagnostics.  A key still registered in ``_spawning``
        has not reached its caller yet, so it is not ours to reclaim.
        """
        with self._state_lock:
            candidates = [
                (key, self._last_used.get(key, 0.0))
                for key in self._clients
                if self._inflight.get(key, 0) == 0 and key not in self._spawning
            ]
        candidates.sort(key=lambda kv: kv[1])
        return candidates

    async def _evict(
        self, key: Tuple[str, str], reason: str, *, handoff: Optional[float] = None
    ) -> bool:
        """Shut down the client at *key* and drop its bookkeeping.

        The bookkeeping pop is synchronous; only the process shutdown is
        awaited, and *handoff* decides how much of it the caller pays for.
        ``None`` — the default, used by the hand-crank and the detached
        sweeps — waits for the shutdown to finish.  A float waits at most
        that long and then leaves the rest running, with the slot still
        reserved in ``_shutting_down``.  That bound is what keeps a wedged
        victim off a request's critical path.

        Returns False if it vanished under us (a concurrent shutdown, or
        a request that started between the scan and the pop) so callers
        can keep their evicted-list honest.
        """
        with self._state_lock:
            if self._inflight.get(key, 0) > 0:
                return False
            client = self._clients.pop(key, None)
            self._last_used.pop(key, None)
        if client is None:
            return False
        eventlog.log_evicted(key[0], key[1], reason)
        # Reserve the slot the moment the task exists.  ``create_task``
        # only schedules — the drain cannot have run and released the
        # reservation before the registration below, because nothing
        # awaits in between.
        task = asyncio.create_task(self._drain_shutdown(key, client))
        with self._state_lock:
            self._shutting_down[task] = key
        if handoff is None:
            await asyncio.shield(task)
        elif handoff > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=handoff)
            except asyncio.TimeoutError:
                logger.debug(
                    "evicted client %s is still shutting down after %.2fs; "
                    "leaving it to drain with its slot reserved",
                    key,
                    handoff,
                )
        return True

    async def _drain_shutdown(self, key: Tuple[str, str], client: LSPClient) -> None:
        """Drive an evicted client's shutdown and free its reservation.

        Always runs to completion even when the caller stopped waiting at
        its handoff bound — a detached shutdown that nobody finishes would
        orphan the process the cap exists to reclaim.  Failures are logged
        rather than swallowed: nobody calls ``.result()`` on this task.

        Releases its OWN reservation by task identity, so a concurrent
        shutdown of the same workspace keeps its slot reserved instead of
        being dropped by whichever drain happens to finish first.
        """
        try:
            await client.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("evicted client %s failed to shut down cleanly: %s", key, e)
        finally:
            # Always a Task here — this coroutine only ever runs via
            # ``create_task`` in ``_evict`` — but ``current_task()`` is
            # typed Optional, and silently skipping the release would
            # leak the slot rather than surface the impossible case.
            task = asyncio.current_task()
            with self._state_lock:
                if task is not None:
                    self._shutting_down.pop(task, None)

    async def _enforce_cap_async(
        self,
        protect: Optional[Tuple[str, str]] = None,
        *,
        handoff: Optional[float] = None,
    ) -> List[Tuple[str, str]]:
        """Drain to ``_max_clients``, evicting least-recently-used first.

        *protect* is the key that just spawned: evicting the client the
        caller is about to use would turn the cap into an infinite
        spawn/evict loop.

        A key registered in ``_spawning`` but not yet in ``_clients``
        counts against the cap.  It is a server that is about to exist,
        and the spawn path enforces the cap *before* ``client.start()``,
        so that reservation is what keeps the fleet off ``cap + 1`` live
        servers.  It also stops concurrent spawns for different roots
        from each claiming the same single free slot.
        """
        evicted: List[Tuple[str, str]] = []
        # ONE deadline for the whole sweep, not a fresh bound per victim.
        # *handoff* is the caller's guard band — the slice of its request
        # budget it is willing to spend making room.  Charging it once per
        # victim made the cost scale with fleet overage: four wedged
        # victims spent 4 x 0.5s before the replacement even started, so
        # ``get_diagnostics_sync`` could still time out before
        # ``wait_for_diagnostics`` got its configured budget — the silent
        # no-diagnostics result this bound exists to prevent.  Past the
        # deadline eviction continues at zero wait: the shutdowns detach
        # and keep their slots reserved, which is the same trade ``_evict``
        # already makes for a single wedged victim.
        deadline = None if handoff is None else time.monotonic() + handoff
        while True:
            with self._state_lock:
                overage = self._overage_locked()
                residual, reason = self._binding_overage_locked(
                    exclude_draining=True
                )
            if overage <= 0:
                break
            if residual <= 0:
                # Every slot still over the cap already has a shutdown
                # running against it.  Evicting another would drain the
                # fleet past what the overage justifies — the shutdowns
                # in flight are what close this gap, and each one frees
                # its reservation when its process is actually gone.
                # (``handoff=None`` awaits completion, so this is only
                # ever reachable from a bounded-handoff caller.)
                #
                # Asked as "would we still be over once the drains land?"
                # rather than "are there at least as many drains as
                # victims?".  Those agree only while every server costs
                # the same.  With per-type footprints they do not: a
                # 150 MiB yaml shutdown is one drain against a one-victim
                # overage, so the count test stopped the sweep — and once
                # that cheap process exited, three 1.7 GiB typescript
                # servers were left over a 4 GiB budget with no sweep
                # scheduled to notice.  Comparing bytes-after-drain keeps
                # the original intent and is exactly equivalent whenever
                # the fleet is homogeneous.
                break
            victim = next(
                (key for key, _ in self._evictable() if key != protect),
                None,
            )
            if victim is None:
                # Everything left is in-flight or protected.  Going over
                # the cap briefly is the right trade against killing a
                # live request.  ``_release`` re-runs this sweep the
                # moment the first of them goes idle, so "briefly" is
                # enforced rather than hoped for — the idle reaper is
                # not the backstop here, and at ``idle_timeout: 0`` it
                # does not run at all.
                logger.debug(
                    "LSP %s by %d but all clients are busy",
                    reason,
                    residual,
                )
                break
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if await self._evict(victim, reason, handoff=remaining):
                evicted.append(victim)
            elif victim in self._clients:
                # It became busy between the scan and the pop, and is
                # still here — stop rather than spin on the same key.
                break
        return evicted

    def enforce_cap_now(self) -> List[Tuple[str, str]]:
        """Evict least-recently-used clients until the cap is satisfied.

        The spawn path enforces the cap on its own; this is the
        hand-crank for tests and ``hermes lsp`` tooling.
        """
        if not self._enabled:
            return []
        try:
            return self._loop.run(self._enforce_cap_async(), timeout=15.0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP cap enforcement failed: %s", e)
            return []

    def _touch(self, client: LSPClient) -> None:
        """Refresh the last-used timestamp for a client we just used.

        Guarded on membership so a reaped-mid-operation client can't
        resurrect an orphan ``_last_used`` entry after the reaper popped
        the key.  All writers and the reaper run on the background loop
        thread; the lock keeps this consistent with the reader anyway.
        """
        key = (client.server_id, client.workspace_root)
        with self._state_lock:
            if key in self._clients:
                self._last_used[key] = _idle_clock()

    async def _idle_reaper_loop(self) -> None:
        interval = min(60.0, self._idle_timeout)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._reap_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # A transient sweep error must not kill the reaper —
                # otherwise one bad shutdown permanently re-opens the
                # unbounded-accumulation leak this loop exists to fix.
                logger.debug("LSP idle reaper sweep error: %s", e)

    async def _reap_idle_once(self) -> None:
        cutoff = _idle_clock() - self._idle_timeout
        with self._state_lock:
            # The in-flight guard closes the residual race the
            # MIN_IDLE_TIMEOUT clamp only makes unlikely: reaping a client
            # mid-request makes the outer wait time out, and that handler
            # marks the pair broken for the whole process lifetime.
            idle_keys = [
                key
                for key in self._clients
                if self._last_used.get(key, 0) < cutoff
                and self._inflight.get(key, 0) == 0
            ]
            clients = [self._clients.pop(key) for key in idle_keys]
            for key in idle_keys:
                self._last_used.pop(key, None)
        if clients:
            eventlog.log_reaped(
                [(c.server_id, c.workspace_root) for c in clients],
                self._idle_timeout,
            )
            await asyncio.gather(
                *(client.shutdown() for client in clients),
                return_exceptions=True,
            )

    async def _shutdown_async(self) -> None:
        reaper = self._idle_reaper_task
        self._idle_reaper_task = None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
        with self._state_lock:
            clients = list(self._clients.values())
            # Evictions whose handoff bound expired are still draining.
            # Stopping the loop out from under them would orphan exactly
            # the processes this shutdown exists to reclaim.
            draining = list(self._shutting_down)
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
            self._inflight.clear()
        await asyncio.gather(
            *(c.shutdown() for c in clients),
            *draining,
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command.

        Carries the bounds in force alongside the fleet.  ``hermes lsp
        status`` is how an operator audits a host that is pinned at its
        cap; without the limits and the per-client ages, the only way to
        learn what the cap *is* is to read this module and re-derive it
        from host RAM.
        """
        # Must be the same clock that writes ``_last_used`` (see
        # ``_touch`` and ``_reap_idle_once``) — these values are only
        # meaningful relative to each other, so reading a different
        # source here turns every age into the gap between two epochs.
        now = _idle_clock()
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                    "idle_seconds": max(0.0, now - self._last_used.get(k, now)),
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
