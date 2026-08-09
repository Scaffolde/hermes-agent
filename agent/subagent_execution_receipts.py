"""Execution receipt state machine for separate-process subagent launches.

Phase 2 boundary: these receipts attest what the host OBSERVED about a
child process execution (backend, root pid, times, exit/signal, broker
transcript digest, cleanup, containment evidence).  They are distinct
from Phase 1 launch receipts, which attest in-process policy only, and
never relabel them.

Legal transitions::

    CREATED -> STARTED -> {SUCCEEDED, FAILED, CANCELLED,
                           TIMED_OUT, CONTAINMENT_FAILED}

Terminal states admit no further transitions and STARTED cannot be
skipped.  The recorder is thread safe; every snapshot is an immutable
frozen dataclass whose canonical hash is recomputable from ``to_dict``.
Receipts structurally exclude secrets: there is no field for key
material, and only digests, labels, and diagnostics are accepted.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re
import secrets
import threading
import time
from typing import Any, Optional

from agent.subagent_broker_protocol import canonical_json, _sha256_hex

EXECUTION_RECEIPT_VERSION = 1

_MAX_LABEL_CHARS = 200
_MAX_ENTRY_CHARS = 4_000
_MAX_ENTRIES = 64
_MAX_EXECUTION_ID_CHARS = 128
_MAX_EXIT_CODE = 2**31
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutionReceiptError(ValueError):
    """A receipt transition or field is invalid; the recorder is unchanged."""


class SubagentExecutionState(str, enum.Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    CONTAINMENT_FAILED = "CONTAINMENT_FAILED"


TERMINAL_EXECUTION_STATES = frozenset(
    {
        SubagentExecutionState.SUCCEEDED,
        SubagentExecutionState.FAILED,
        SubagentExecutionState.CANCELLED,
        SubagentExecutionState.TIMED_OUT,
        SubagentExecutionState.CONTAINMENT_FAILED,
    }
)


def _require_hex_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST_RE.match(value):
        raise ExecutionReceiptError(
            f"{field} must be a 64-char lowercase hex digest."
        )
    return value


def _require_label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_LABEL_CHARS
    ):
        raise ExecutionReceiptError(
            f"{field} must be a non-empty string of at most "
            f"{_MAX_LABEL_CHARS} characters."
        )
    return value


def _require_entry_tuple(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ExecutionReceiptError(f"{field} must be a sequence of strings.")
    if len(value) > _MAX_ENTRIES:
        raise ExecutionReceiptError(
            f"{field} must contain at most {_MAX_ENTRIES} entries."
        )
    entries = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > _MAX_ENTRY_CHARS
        ):
            raise ExecutionReceiptError(
                f"{field} entries must be non-empty strings of at most "
                f"{_MAX_ENTRY_CHARS} characters."
            )
        entries.append(item)
    return tuple(entries)


def _require_timestamp(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ExecutionReceiptError(f"{field} must be a finite positive number.")
    return float(value)


def _optional_bounded_int(value: Any, field: str, low: int, high: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ExecutionReceiptError(
            f"{field} must be an integer in [{low}, {high}]."
        )
    return value


@dataclasses.dataclass(frozen=True)
class SubagentExecutionReceipt:
    """Immutable snapshot of one execution's host-observed state."""

    receipt_version: int
    execution_id: str
    launch_receipt_digest: str
    backend: str
    containment_mode: str
    state: SubagentExecutionState
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    root_pid: Optional[int]
    exit_code: Optional[int]
    term_signal: Optional[int]
    broker_transcript_digest: Optional[str]
    requested_cleanup: tuple[str, ...]
    observed_cleanup: tuple[str, ...]
    containment_evidence: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "execution_id": self.execution_id,
            "launch_receipt_digest": self.launch_receipt_digest,
            "backend": self.backend,
            "containment_mode": self.containment_mode,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "root_pid": self.root_pid,
            "exit_code": self.exit_code,
            "term_signal": self.term_signal,
            "broker_transcript_digest": self.broker_transcript_digest,
            "requested_cleanup": list(self.requested_cleanup),
            "observed_cleanup": list(self.observed_cleanup),
            "containment_evidence": list(self.containment_evidence),
            "diagnostics": list(self.diagnostics),
        }

    def canonical_hash(self) -> str:
        """Deterministic SHA-256 over the canonical-JSON receipt payload."""
        return _sha256_hex(canonical_json(self.to_dict()))


class SubagentExecutionRecorder:
    """Thread-safe execution state machine emitting immutable receipts."""

    def __init__(
        self,
        *,
        launch_receipt_digest: str,
        backend: str,
        containment_mode: str,
        requested_cleanup: Any = (),
        execution_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> None:
        self._launch_receipt_digest = _require_hex_digest(
            launch_receipt_digest, "launch_receipt_digest"
        )
        self._backend = _require_label(backend, "backend")
        self._containment_mode = _require_label(containment_mode, "containment_mode")
        self._requested_cleanup = _require_entry_tuple(
            requested_cleanup, "requested_cleanup"
        )
        if execution_id is None:
            execution_id = secrets.token_hex(16)
        elif (
            not isinstance(execution_id, str)
            or not execution_id.strip()
            or len(execution_id) > _MAX_EXECUTION_ID_CHARS
        ):
            raise ExecutionReceiptError(
                "execution_id must be a non-empty string of at most "
                f"{_MAX_EXECUTION_ID_CHARS} characters."
            )
        self._execution_id = execution_id
        self._created_at = (
            time.time()
            if created_at is None
            else _require_timestamp(created_at, "created_at")
        )
        self._lock = threading.Lock()
        self._state = SubagentExecutionState.CREATED
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._root_pid: Optional[int] = None
        self._exit_code: Optional[int] = None
        self._term_signal: Optional[int] = None
        self._broker_transcript_digest: Optional[str] = None
        self._observed_cleanup: tuple[str, ...] = ()
        self._containment_evidence: tuple[str, ...] = ()
        self._diagnostics: tuple[str, ...] = ()

    def snapshot(self) -> SubagentExecutionReceipt:
        with self._lock:
            return self._snapshot_locked()

    def mark_started(
        self, *, root_pid: Any, started_at: Optional[float] = None
    ) -> SubagentExecutionReceipt:
        pid = _optional_bounded_int(root_pid, "root_pid", 1, 2**31)
        if pid is None:
            raise ExecutionReceiptError("root_pid is required to mark STARTED.")
        stamp = (
            time.time()
            if started_at is None
            else _require_timestamp(started_at, "started_at")
        )
        with self._lock:
            if self._state is not SubagentExecutionState.CREATED:
                raise ExecutionReceiptError(
                    f"Cannot mark STARTED from {self._state.value}; only a "
                    "CREATED execution may start."
                )
            if stamp < self._created_at:
                raise ExecutionReceiptError("started_at must be after created_at.")
            self._state = SubagentExecutionState.STARTED
            self._root_pid = pid
            self._started_at = stamp
            return self._snapshot_locked()

    def mark_succeeded(self, **evidence: Any) -> SubagentExecutionReceipt:
        return self._complete(SubagentExecutionState.SUCCEEDED, **evidence)

    def mark_failed(self, **evidence: Any) -> SubagentExecutionReceipt:
        return self._complete(SubagentExecutionState.FAILED, **evidence)

    def mark_cancelled(self, **evidence: Any) -> SubagentExecutionReceipt:
        return self._complete(SubagentExecutionState.CANCELLED, **evidence)

    def mark_timed_out(self, **evidence: Any) -> SubagentExecutionReceipt:
        return self._complete(SubagentExecutionState.TIMED_OUT, **evidence)

    def mark_containment_failed(self, **evidence: Any) -> SubagentExecutionReceipt:
        return self._complete(
            SubagentExecutionState.CONTAINMENT_FAILED,
            allow_created=True,
            **evidence,
        )

    def _complete(
        self,
        state: SubagentExecutionState,
        *,
        exit_code: Any = None,
        term_signal: Any = None,
        broker_transcript_digest: Any = None,
        observed_cleanup: Any = (),
        containment_evidence: Any = (),
        diagnostics: Any = (),
        completed_at: Optional[float] = None,
        allow_created: bool = False,
    ) -> SubagentExecutionReceipt:
        # Validate every field before taking the lock: a rejected transition
        # must leave the recorder byte-for-byte unchanged.
        exit_code = _optional_bounded_int(
            exit_code, "exit_code", -_MAX_EXIT_CODE, _MAX_EXIT_CODE
        )
        term_signal = _optional_bounded_int(term_signal, "term_signal", 1, 128)
        if broker_transcript_digest is not None:
            broker_transcript_digest = _require_hex_digest(
                broker_transcript_digest, "broker_transcript_digest"
            )
        observed_cleanup = _require_entry_tuple(observed_cleanup, "observed_cleanup")
        containment_evidence = _require_entry_tuple(
            containment_evidence, "containment_evidence"
        )
        diagnostics = _require_entry_tuple(diagnostics, "diagnostics")
        stamp = (
            time.time()
            if completed_at is None
            else _require_timestamp(completed_at, "completed_at")
        )
        with self._lock:
            if self._state in TERMINAL_EXECUTION_STATES:
                raise ExecutionReceiptError(
                    f"Execution is terminal ({self._state.value}); receipts "
                    "admit no further transitions."
                )
            if self._state is SubagentExecutionState.CREATED and allow_created:
                pass
            elif self._state is not SubagentExecutionState.STARTED:
                raise ExecutionReceiptError(
                    f"Cannot mark {state.value} from {self._state.value}; the "
                    "STARTED transition cannot be skipped."
                )
            if stamp < self._created_at or (
                self._started_at is not None and stamp < self._started_at
            ):
                raise ExecutionReceiptError(
                    "completed_at must be after created_at and started_at."
                )
            self._state = state
            self._completed_at = stamp
            self._exit_code = exit_code
            self._term_signal = term_signal
            self._broker_transcript_digest = broker_transcript_digest
            self._observed_cleanup = observed_cleanup
            self._containment_evidence = containment_evidence
            self._diagnostics = diagnostics
            return self._snapshot_locked()

    def _snapshot_locked(self) -> SubagentExecutionReceipt:
        return SubagentExecutionReceipt(
            receipt_version=EXECUTION_RECEIPT_VERSION,
            execution_id=self._execution_id,
            launch_receipt_digest=self._launch_receipt_digest,
            backend=self._backend,
            containment_mode=self._containment_mode,
            state=self._state,
            created_at=self._created_at,
            started_at=self._started_at,
            completed_at=self._completed_at,
            root_pid=self._root_pid,
            exit_code=self._exit_code,
            term_signal=self._term_signal,
            broker_transcript_digest=self._broker_transcript_digest,
            requested_cleanup=self._requested_cleanup,
            observed_cleanup=self._observed_cleanup,
            containment_evidence=self._containment_evidence,
            diagnostics=self._diagnostics,
        )
