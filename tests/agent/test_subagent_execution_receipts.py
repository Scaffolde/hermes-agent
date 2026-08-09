"""Focused tests for the subagent execution receipt state machine (Lane A).

Execution receipts are immutable snapshots of a thread-safe recorder.
Legal transitions are CREATED -> STARTED -> {SUCCEEDED, FAILED,
CANCELLED, TIMED_OUT, CONTAINMENT_FAILED}; terminal states admit no
further transitions and STARTED cannot be skipped.  The receipt hash is
canonical and the receipt carries no secret material.
"""

import dataclasses
import hashlib
import threading

import pytest

from agent.subagent_broker_protocol import (
    BrokerGrant,
    SubagentBroker,
    canonical_json,
)
from agent.subagent_execution_receipts import (
    EXECUTION_RECEIPT_VERSION,
    TERMINAL_EXECUTION_STATES,
    ExecutionReceiptError,
    SubagentExecutionRecorder,
    SubagentExecutionState,
)

LAUNCH_DIGEST = hashlib.sha256(b"launch-receipt").hexdigest()

_TERMINAL_MARKS = {
    SubagentExecutionState.SUCCEEDED: "mark_succeeded",
    SubagentExecutionState.FAILED: "mark_failed",
    SubagentExecutionState.CANCELLED: "mark_cancelled",
    SubagentExecutionState.TIMED_OUT: "mark_timed_out",
    SubagentExecutionState.CONTAINMENT_FAILED: "mark_containment_failed",
}


def _recorder(**overrides):
    kwargs: dict = {
        "launch_receipt_digest": LAUNCH_DIGEST,
        "backend": "portable-process",
        "containment_mode": "portable-process-unconfined",
        "requested_cleanup": ("terminate-process-group", "reap-descendants"),
    }
    kwargs.update(overrides)
    return SubagentExecutionRecorder(**kwargs)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_initial_snapshot_records_launch_and_backend_facts():
    recorder = _recorder()
    receipt = recorder.snapshot()
    assert receipt.receipt_version == EXECUTION_RECEIPT_VERSION
    assert receipt.state is SubagentExecutionState.CREATED
    assert receipt.launch_receipt_digest == LAUNCH_DIGEST
    assert receipt.backend == "portable-process"
    assert receipt.containment_mode == "portable-process-unconfined"
    assert receipt.requested_cleanup == (
        "terminate-process-group",
        "reap-descendants",
    )
    assert receipt.execution_id
    assert receipt.created_at > 0
    assert receipt.started_at is None
    assert receipt.completed_at is None
    assert receipt.root_pid is None
    assert receipt.exit_code is None
    assert receipt.term_signal is None
    assert receipt.broker_transcript_digest is None
    assert receipt.observed_cleanup == ()
    assert receipt.containment_evidence == ()
    assert receipt.diagnostics == ()


def test_execution_ids_are_unique_per_recorder():
    assert _recorder().snapshot().execution_id != _recorder().snapshot().execution_id


def test_snapshots_are_immutable():
    receipt = _recorder().snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.state = SubagentExecutionState.SUCCEEDED
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.diagnostics = ("injected",)


def test_earlier_snapshot_is_unaffected_by_later_transitions():
    recorder = _recorder()
    created = recorder.snapshot()
    recorder.mark_started(root_pid=4242)
    assert created.state is SubagentExecutionState.CREATED
    assert created.root_pid is None
    assert recorder.snapshot().state is SubagentExecutionState.STARTED


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------


def test_started_records_root_pid_and_start_time():
    recorder = _recorder()
    receipt = recorder.mark_started(root_pid=4242)
    assert receipt.state is SubagentExecutionState.STARTED
    assert receipt.root_pid == 4242
    assert receipt.started_at is not None
    assert receipt.started_at >= receipt.created_at


@pytest.mark.parametrize("state", sorted(_TERMINAL_MARKS, key=lambda s: s.value))
def test_every_terminal_state_is_reachable_from_started(state):
    recorder = _recorder()
    recorder.mark_started(root_pid=4242)
    receipt = getattr(recorder, _TERMINAL_MARKS[state])(
        broker_transcript_digest=hashlib.sha256(b"transcript").hexdigest(),
        observed_cleanup=("terminate-process-group",),
        containment_evidence=("session-created",),
        diagnostics=("detail",),
    )
    assert receipt.state is state
    assert state in TERMINAL_EXECUTION_STATES
    assert receipt.completed_at is not None
    assert receipt.broker_transcript_digest == hashlib.sha256(b"transcript").hexdigest()
    assert receipt.observed_cleanup == ("terminate-process-group",)
    assert receipt.containment_evidence == ("session-created",)
    assert receipt.diagnostics == ("detail",)


def test_success_records_exit_code_and_failure_records_signal():
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    assert recorder.mark_succeeded(exit_code=0).exit_code == 0

    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    receipt = recorder.mark_failed(term_signal=9)
    assert receipt.term_signal == 9
    assert receipt.exit_code is None


# ---------------------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mark",
    sorted(set(_TERMINAL_MARKS.values()) - {"mark_containment_failed"}),
)
def test_started_cannot_be_skipped(mark):
    recorder = _recorder()
    with pytest.raises(ExecutionReceiptError):
        getattr(recorder, mark)()
    assert recorder.snapshot().state is SubagentExecutionState.CREATED


def test_pre_spawn_containment_failure_has_no_fake_pid():
    receipt = _recorder().mark_containment_failed(
        diagnostics=("strict prerequisites unavailable",)
    )
    assert receipt.state is SubagentExecutionState.CONTAINMENT_FAILED
    assert receipt.root_pid is None
    assert receipt.started_at is None
    assert receipt.completed_at >= receipt.created_at


def test_started_cannot_be_marked_twice():
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_started(root_pid=2)
    assert recorder.snapshot().root_pid == 1


def test_terminal_states_admit_no_further_transitions():
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    recorder.mark_succeeded(exit_code=0)
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_started(root_pid=2)
    for mark in _TERMINAL_MARKS.values():
        with pytest.raises(ExecutionReceiptError):
            getattr(recorder, mark)()
    assert recorder.snapshot().state is SubagentExecutionState.SUCCEEDED


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def test_canonical_hash_is_recomputable_from_the_snapshot():
    recorder = _recorder()
    recorder.mark_started(root_pid=77)
    receipt = recorder.mark_succeeded(exit_code=0, observed_cleanup=("reap",))
    expected = hashlib.sha256(
        canonical_json(receipt.to_dict()).encode("utf-8")
    ).hexdigest()
    assert receipt.canonical_hash() == expected
    assert len(receipt.canonical_hash()) == 64


def test_canonical_hash_is_stable_and_state_sensitive():
    recorder = _recorder()
    created_hash = recorder.snapshot().canonical_hash()
    assert recorder.snapshot().canonical_hash() == created_hash
    started_hash = recorder.mark_started(root_pid=1).canonical_hash()
    assert started_hash != created_hash


def test_receipt_carries_no_broker_secret():
    broker = SubagentBroker(
        launch_receipt_digest=LAUNCH_DIGEST,
        grant=BrokerGrant(
            operations=frozenset({"read_file"}), workspace_root="/tmp/ws"
        ),
    )
    broker.validate(broker.client_signer().sign("read_file", None))
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    receipt = recorder.mark_succeeded(
        exit_code=0, broker_transcript_digest=broker.transcript_digest()
    )
    serialized = canonical_json(receipt.to_dict())
    assert broker.reveal_secret_for_transport().hex() not in serialized
    assert receipt.broker_transcript_digest == broker.transcript_digest()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_recorder_rejects_malformed_launch_digest():
    with pytest.raises(ExecutionReceiptError):
        _recorder(launch_receipt_digest="not-a-digest")


def test_recorder_rejects_blank_backend_and_mode():
    with pytest.raises(ExecutionReceiptError):
        _recorder(backend="")
    with pytest.raises(ExecutionReceiptError):
        _recorder(containment_mode="   ")


@pytest.mark.parametrize("root_pid", [0, -4, True, "42", 4.2])
def test_started_rejects_invalid_root_pids(root_pid):
    recorder = _recorder()
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_started(root_pid=root_pid)
    assert recorder.snapshot().state is SubagentExecutionState.CREATED


def test_terminal_marks_reject_malformed_evidence():
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_failed(diagnostics=("", "empty entry"))
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_failed(observed_cleanup=(42,))
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_failed(broker_transcript_digest="zz")
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_failed(diagnostics=tuple(f"d{i}" for i in range(100)))
    with pytest.raises(ExecutionReceiptError):
        recorder.mark_failed(exit_code=True)
    # The recorder is still usable after rejected evidence: fail closed,
    # not wedged.
    assert recorder.mark_failed(exit_code=1).state is SubagentExecutionState.FAILED


def test_recorder_rejects_malformed_requested_cleanup():
    with pytest.raises(ExecutionReceiptError):
        _recorder(requested_cleanup=("ok", ""))
    with pytest.raises(ExecutionReceiptError):
        _recorder(requested_cleanup="not-a-tuple")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_exactly_one_terminal_transition_wins_under_contention():
    recorder = _recorder()
    recorder.mark_started(root_pid=1)
    wins = []
    losses = []
    barrier = threading.Barrier(16)

    def attempt(index):
        barrier.wait()
        try:
            if index % 2:
                recorder.mark_succeeded(exit_code=0)
            else:
                recorder.mark_failed(exit_code=1)
        except ExecutionReceiptError:
            losses.append(index)
        else:
            wins.append(index)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(wins) == 1
    assert len(losses) == 15
    assert recorder.snapshot().state in TERMINAL_EXECUTION_STATES
