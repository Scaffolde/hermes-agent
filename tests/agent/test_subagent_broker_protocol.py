"""Focused tests for the authenticated subagent broker protocol (Lane A).

The broker is host-owned: a random per-launch capability secret signs
canonical-JSON envelopes with HMAC-SHA256.  Validation is fail-closed:
exact protocol-version / capability / launch-digest match, strictly
increasing sequences with replay rejection, grant-checked operations,
and closed or revoked capabilities reject every operation.  The secret
is never serialized.
"""

import hashlib
import hmac
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.subagent_broker_protocol import (
    BROKER_PROTOCOL_VERSION,
    BrokerEnvelopeRejected,
    BrokerEnvelopeSigner,
    BrokerGrant,
    BrokerProtocolError,
    SubagentBroker,
    canonical_json,
)

LAUNCH_DIGEST = hashlib.sha256(b"launch-receipt").hexdigest()
OTHER_DIGEST = hashlib.sha256(b"other-launch-receipt").hexdigest()

_AUTHORITY_KEYS = (
    "protocol_version",
    "capability_id",
    "launch_receipt_digest",
    "sequence",
    "operation",
    "body_digest",
)


def _grant(**overrides):
    kwargs: dict = {
        "operations": frozenset({"read_file", "write_file"}),
        "workspace_root": "/tmp/hermes-broker-ws",
    }
    kwargs.update(overrides)
    return BrokerGrant(**kwargs)


def _broker(**overrides):
    kwargs: dict = {"launch_receipt_digest": LAUNCH_DIGEST, "grant": _grant()}
    kwargs.update(overrides)
    return SubagentBroker(**kwargs)


def _resign(envelope, secret, **changes):
    """Rebuild a valid MAC after mutating authority fields (attacker-with-secret)."""
    env = dict(envelope)
    env.update(changes)
    authority = {key: env[key] for key in _AUTHORITY_KEYS}
    env["mac"] = hmac.new(
        secret, canonical_json(authority).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return env


def _reason(excinfo) -> str:
    return excinfo.value.reason_code


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_signed_envelope_validates_with_grant_authority():
    broker = _broker()
    signer = broker.client_signer()
    envelope = signer.sign("read_file", {"path": "notes.md"})
    request = broker.validate(envelope)
    assert request.operation == "read_file"
    assert request.sequence == 1
    assert request.body == {"path": "notes.md"}
    assert request.capability_id == broker.capability_id
    assert request.workspace_root == "/tmp/hermes-broker-ws"


def test_sequences_increase_strictly_across_envelopes():
    broker = _broker()
    signer = broker.client_signer()
    first = broker.validate(signer.sign("read_file", None))
    second = broker.validate(signer.sign("write_file", {"path": "a", "data": "b"}))
    assert (first.sequence, second.sequence) == (1, 2)


def test_capability_id_is_opaque_and_per_launch():
    broker_a = _broker()
    broker_b = _broker()
    assert broker_a.capability_id != broker_b.capability_id
    assert broker_a.capability_id != LAUNCH_DIGEST
    assert len(broker_a.capability_id) >= 32
    assert all(ch in "0123456789abcdef" for ch in broker_a.capability_id)


def test_envelope_is_canonical_json_serializable_and_carries_no_secret():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", {"path": "x"})
    secret = broker.reveal_secret_for_transport()
    serialized = canonical_json(envelope)
    assert secret.hex() not in serialized
    assert canonical_json(envelope) == serialized  # deterministic


# ---------------------------------------------------------------------------
# Authenticity and integrity rejection
# ---------------------------------------------------------------------------


def test_wrong_secret_mac_is_rejected():
    broker = _broker()
    forged_signer = BrokerEnvelopeSigner(
        capability_id=broker.capability_id,
        secret=b"f" * 32,
        launch_receipt_digest=LAUNCH_DIGEST,
    )
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(forged_signer.sign("read_file", None))
    assert _reason(excinfo) == "mac-invalid"


def test_tampered_mac_of_correct_length_is_rejected():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", None)
    envelope["mac"] = "0" * 64
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "mac-invalid"


def test_tampered_body_with_original_digest_is_rejected():
    broker = _broker()
    envelope = broker.client_signer().sign("write_file", {"path": "a", "data": "ok"})
    envelope["body"] = {"path": "a", "data": "evil"}
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "body-digest-mismatch"


def test_tampered_body_digest_breaks_the_mac():
    broker = _broker()
    envelope = broker.client_signer().sign("write_file", {"path": "a", "data": "ok"})
    envelope["body_digest"] = hashlib.sha256(b"evil").hexdigest()
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "mac-invalid"


def test_protocol_version_must_match_exactly():
    broker = _broker()
    secret = broker.reveal_secret_for_transport()
    envelope = broker.client_signer().sign("read_file", None)
    resigned = _resign(envelope, secret, protocol_version=BROKER_PROTOCOL_VERSION + 1)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(resigned)
    assert _reason(excinfo) == "protocol-version-mismatch"


def test_capability_id_must_match_exactly():
    broker = _broker()
    secret = broker.reveal_secret_for_transport()
    envelope = broker.client_signer().sign("read_file", None)
    resigned = _resign(envelope, secret, capability_id="deadbeef" * 4)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(resigned)
    assert _reason(excinfo) == "capability-mismatch"


def test_launch_receipt_digest_must_match_exactly():
    broker = _broker()
    secret = broker.reveal_secret_for_transport()
    envelope = broker.client_signer().sign("read_file", None)
    resigned = _resign(envelope, secret, launch_receipt_digest=OTHER_DIGEST)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(resigned)
    assert _reason(excinfo) == "launch-digest-mismatch"


# ---------------------------------------------------------------------------
# Sequence and replay rejection
# ---------------------------------------------------------------------------


def test_replayed_envelope_is_rejected():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", None)
    broker.validate(envelope)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "sequence-rejected"


def test_lower_sequence_after_higher_is_rejected():
    broker = _broker()
    signer = broker.client_signer()
    first = signer.sign("read_file", None)
    second = signer.sign("read_file", None)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(second)
    assert _reason(excinfo) == "sequence-rejected"
    broker.validate(first)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(first)
    assert _reason(excinfo) == "sequence-rejected"


def test_sequence_zero_is_rejected():
    broker = _broker()
    secret = broker.reveal_secret_for_transport()
    envelope = broker.client_signer().sign("read_file", None)
    resigned = _resign(envelope, secret, sequence=0)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(resigned)
    assert _reason(excinfo) == "sequence-rejected"


def test_rejected_envelope_does_not_advance_the_sequence():
    broker = _broker()
    signer = broker.client_signer()
    envelope = signer.sign("read_file", None)
    tampered = dict(envelope)
    tampered["mac"] = "0" * 64
    with pytest.raises(BrokerEnvelopeRejected):
        broker.validate(tampered)
    # The untampered original still validates: rejection left no state behind.
    assert broker.validate(envelope).sequence == 1


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------


def test_operation_outside_grant_is_rejected():
    broker = _broker()
    envelope = broker.client_signer().sign("delete_file", None)
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "operation-not-granted"


def test_grant_requires_at_least_one_operation():
    with pytest.raises(BrokerProtocolError):
        _grant(operations=frozenset())


def test_grant_rejects_malformed_operation_names():
    with pytest.raises(BrokerProtocolError):
        _grant(operations=frozenset({"Read File"}))


def test_grant_requires_absolute_workspace_root():
    with pytest.raises(BrokerProtocolError):
        _grant(workspace_root="relative/workspace")


# ---------------------------------------------------------------------------
# Close / revoke fail closed
# ---------------------------------------------------------------------------


def test_closed_capability_rejects_every_operation():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", None)
    broker.close("launch complete")
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "capability-closed"
    assert broker.closed


def test_revoked_capability_rejects_every_operation():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", None)
    broker.revoke("policy violation")
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(envelope)
    assert _reason(excinfo) == "capability-revoked"
    assert broker.revoked


def test_revocation_wins_over_close():
    broker = _broker()
    broker.close("done")
    broker.revoke("compromise suspected")
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(broker.client_signer().sign("read_file", None))
    assert _reason(excinfo) == "capability-revoked"


def test_close_and_revoke_require_a_reason():
    broker = _broker()
    with pytest.raises(BrokerProtocolError):
        broker.close("")
    with pytest.raises(BrokerProtocolError):
        broker.revoke("   ")


# ---------------------------------------------------------------------------
# Malformed envelopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: "not-a-mapping",
        lambda env: {k: v for k, v in env.items() if k != "mac"},
        lambda env: {**env, "extra": 1},
        lambda env: {**env, "sequence": True},
        lambda env: {**env, "sequence": "1"},
        lambda env: {**env, "sequence": -1},
        lambda env: {**env, "operation": 7},
        lambda env: {**env, "operation": "Bad Operation"},
        lambda env: {**env, "mac": "zz"},
        lambda env: {**env, "body_digest": "abc"},
        lambda env: {**env, "launch_receipt_digest": "abc"},
        lambda env: {**env, "protocol_version": "1"},
        lambda env: {**env, "capability_id": ""},
        lambda env: {**env, "body": {1: "non-string-key"}},
        lambda env: {**env, "body": object()},
        lambda env: {**env, "body": float("nan")},
    ],
)
def test_malformed_envelopes_are_rejected(mutate):
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", {"path": "x"})
    with pytest.raises(BrokerEnvelopeRejected) as excinfo:
        broker.validate(mutate(envelope))
    assert _reason(excinfo) == "malformed-envelope"


def test_canonical_json_rejects_unsafe_values():
    for value in (object(), {1: "x"}, float("inf"), {"k": {"n": float("nan")}}):
        with pytest.raises(BrokerProtocolError):
            canonical_json(value)


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


def test_secret_is_random_per_launch_and_long_enough():
    broker_a = _broker()
    broker_b = _broker()
    secret_a = broker_a.reveal_secret_for_transport()
    secret_b = broker_b.reveal_secret_for_transport()
    assert isinstance(secret_a, bytes) and len(secret_a) >= 32
    assert secret_a != secret_b


def test_secret_never_appears_in_repr():
    broker = _broker()
    signer = broker.client_signer()
    secret_hex = broker.reveal_secret_for_transport().hex()
    assert secret_hex not in repr(broker)
    assert secret_hex not in str(broker)
    assert secret_hex not in repr(signer)


def test_broker_and_signer_refuse_pickle_serialization():
    broker = _broker()
    with pytest.raises(TypeError):
        pickle.dumps(broker)
    with pytest.raises(TypeError):
        pickle.dumps(broker.client_signer())


def test_signer_requires_a_strong_secret():
    with pytest.raises(BrokerProtocolError):
        BrokerEnvelopeSigner(
            capability_id="abc123",
            secret=b"short",
            launch_receipt_digest=LAUNCH_DIGEST,
        )


def test_broker_requires_a_valid_launch_receipt_digest():
    with pytest.raises(BrokerProtocolError):
        SubagentBroker(launch_receipt_digest="not-a-digest", grant=_grant())


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def test_transcript_digest_changes_with_broker_events():
    broker = _broker()
    initial = broker.transcript_digest()
    assert len(initial) == 64
    assert broker.transcript_digest() == initial  # stable between events
    broker.validate(broker.client_signer().sign("read_file", None))
    after_accept = broker.transcript_digest()
    assert after_accept != initial
    with pytest.raises(BrokerEnvelopeRejected):
        broker.validate({"junk": True})
    after_reject = broker.transcript_digest()
    assert after_reject != after_accept
    broker.close("launch complete")
    assert broker.transcript_digest() != after_reject


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_signing_yields_unique_strict_sequences():
    broker = _broker()
    signer = broker.client_signer()
    with ThreadPoolExecutor(max_workers=8) as pool:
        envelopes = list(
            pool.map(lambda _: signer.sign("read_file", None), range(200))
        )
    sequences = sorted(env["sequence"] for env in envelopes)
    assert sequences == list(range(1, 201))
    for envelope in sorted(envelopes, key=lambda env: env["sequence"]):
        broker.validate(envelope)


def test_concurrent_validation_accepts_a_replayed_envelope_exactly_once():
    broker = _broker()
    envelope = broker.client_signer().sign("read_file", None)
    accepted = []
    rejected = []
    barrier = threading.Barrier(16)

    def attempt():
        barrier.wait()
        try:
            broker.validate(dict(envelope))
        except BrokerEnvelopeRejected as exc:
            rejected.append(exc.reason_code)
        else:
            accepted.append(True)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(accepted) == 1
    assert len(rejected) == 15
    assert set(rejected) == {"sequence-rejected"}
