"""Authenticated broker protocol for host-owned subagent execution (Phase 2, Lane A).

The host creates one :class:`SubagentBroker` per separate-process launch.
The broker owns a random per-launch capability secret and an opaque
capability ID.  Every request the worker sends is a canonical-JSON
envelope carrying the protocol version, capability ID, launch receipt
digest, a strictly increasing sequence number, the operation name, a
digest of the request body, and an HMAC-SHA256 over all authority
fields.  Validation is fail-closed:

- constant-time MAC comparison with the per-launch secret;
- exact protocol-version, capability, and launch-receipt-digest match;
- strictly increasing sequences (replays and reordering are rejected);
- operations must appear in the host-issued :class:`BrokerGrant`;
- a closed or revoked capability rejects every operation.

The capability secret is never serialized: broker and signer refuse
pickling, redact their reprs, and no envelope, transcript, or receipt
ever contains it.  ``reveal_secret_for_transport`` exists solely so the
integration layer can hand the secret to the worker over an inherited
pipe/socketpair — never via argv, environment, logs, or model context.

Workspace authority note: the grant records the descriptor-owned
workspace root; canonicalizing request paths beneath it is the
integration layer's job (contract: Lane A carries authority, Lane B/
integration enforce the filesystem boundary).
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
from typing import Any, Mapping, NoReturn, Optional

BROKER_PROTOCOL_VERSION = 1

_MIN_SECRET_BYTES = 32
_MAX_REASON_CHARS = 2_000
_MAX_SEQUENCE = 2**63
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
# Printable ASCII only: hmac.compare_digest raises on non-ASCII str input,
# so a fail-closed parse must never let such a value reach the comparison.
_CAPABILITY_ID_RE = re.compile(r"^[\x21-\x7e]{1,128}$")

_AUTHORITY_KEYS = (
    "protocol_version",
    "capability_id",
    "launch_receipt_digest",
    "sequence",
    "operation",
    "body_digest",
)
_ENVELOPE_KEYS = frozenset(_AUTHORITY_KEYS) | {"body", "mac"}


class BrokerProtocolError(ValueError):
    """A broker object cannot be safely constructed or operated."""


class BrokerEnvelopeRejected(BrokerProtocolError):
    """An envelope failed fail-closed validation; ``reason_code`` says why."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        message = f"Broker envelope rejected ({reason_code})."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)
        self.reason_code = reason_code


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, minimal separators, ASCII, no NaN.

    Only JSON-safe trees (str-keyed mappings, sequences, str, bool, int,
    finite float, None) are accepted; anything else raises
    :class:`BrokerProtocolError` instead of being silently coerced.
    """
    _assert_json_safe(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _assert_json_safe(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BrokerProtocolError("Non-finite float in canonical JSON.")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_json_safe(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise BrokerProtocolError(
                    "Canonical JSON mapping keys must be strings."
                )
            _assert_json_safe(item)
        return
    raise BrokerProtocolError(
        f"Unsupported type in canonical JSON: {type(value).__name__}."
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_operation_name(operation: Any) -> str:
    if not isinstance(operation, str) or not _OPERATION_RE.match(operation):
        raise BrokerProtocolError(
            "operation must match ^[a-z][a-z0-9._-]{0,127}$."
        )
    return operation


def _validate_launch_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST_RE.match(value):
        raise BrokerProtocolError(f"{field} must be a 64-char lowercase hex digest.")
    return value


def _envelope_mac(secret: bytes, authority: Mapping[str, Any]) -> str:
    payload = canonical_json({key: authority[key] for key in _AUTHORITY_KEYS})
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclasses.dataclass(frozen=True)
class BrokerGrant:
    """Host-issued authority: exact operation names plus workspace root.

    ``workspace_root`` is recorded authority only — the integration layer
    canonicalizes request paths beneath a descriptor-owned root; the grant
    itself never touches the filesystem.
    """

    operations: frozenset[str]
    workspace_root: str

    def __post_init__(self) -> None:
        raw = self.operations
        if isinstance(raw, str) or not isinstance(raw, (frozenset, set, list, tuple)):
            raise BrokerProtocolError(
                "operations must be a collection of operation names."
            )
        normalized = frozenset(raw)
        if not normalized:
            raise BrokerProtocolError("A grant requires at least one operation.")
        for operation in normalized:
            _validate_operation_name(operation)
        object.__setattr__(self, "operations", normalized)
        root = self.workspace_root
        if (
            not isinstance(root, str)
            or not root.strip()
            or "\x00" in root
            or not os.path.isabs(root)
        ):
            raise BrokerProtocolError(
                "workspace_root must be an absolute path string."
            )


@dataclasses.dataclass(frozen=True)
class ValidatedBrokerRequest:
    """Immutable, host-validated view of one accepted envelope."""

    capability_id: str
    launch_receipt_digest: str
    sequence: int
    operation: str
    body: Any
    body_digest: str
    workspace_root: str


class BrokerEnvelopeSigner:
    """Client-side envelope builder for one capability.

    Holds the capability secret in memory only; refuses serialization and
    redacts its repr.  ``sign`` is thread safe and allocates strictly
    increasing sequence numbers starting at 1.
    """

    def __init__(
        self,
        *,
        capability_id: str,
        secret: bytes,
        launch_receipt_digest: str,
    ) -> None:
        if not isinstance(capability_id, str) or not _CAPABILITY_ID_RE.match(
            capability_id
        ):
            raise BrokerProtocolError(
                "capability_id must be 1-128 printable ASCII characters."
            )
        if not isinstance(secret, bytes) or len(secret) < _MIN_SECRET_BYTES:
            raise BrokerProtocolError(
                f"secret must be at least {_MIN_SECRET_BYTES} bytes."
            )
        self._capability_id = capability_id
        self._secret = secret
        self._launch_receipt_digest = _validate_launch_digest(
            launch_receipt_digest, "launch_receipt_digest"
        )
        self._lock = threading.Lock()
        self._sequence = 0

    def sign(self, operation: str, body: Any = None) -> dict[str, Any]:
        _validate_operation_name(operation)
        body_digest = _sha256_hex(canonical_json(body))
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        authority = {
            "protocol_version": BROKER_PROTOCOL_VERSION,
            "capability_id": self._capability_id,
            "launch_receipt_digest": self._launch_receipt_digest,
            "sequence": sequence,
            "operation": operation,
            "body_digest": body_digest,
        }
        envelope = dict(authority)
        envelope["body"] = body
        envelope["mac"] = _envelope_mac(self._secret, authority)
        return envelope

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def __repr__(self) -> str:
        return (
            f"<BrokerEnvelopeSigner capability_id={self._capability_id!r} "
            "secret=REDACTED>"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "BrokerEnvelopeSigner holds a live capability secret and must "
            "never be serialized."
        )


class SubagentBroker:
    """Host-owned authenticated broker for one separate-process launch."""

    def __init__(
        self, *, launch_receipt_digest: str, grant: BrokerGrant
    ) -> None:
        self._launch_receipt_digest = _validate_launch_digest(
            launch_receipt_digest, "launch_receipt_digest"
        )
        if not isinstance(grant, BrokerGrant):
            raise BrokerProtocolError("grant must be a BrokerGrant.")
        self._grant = grant
        self._capability_id = secrets.token_hex(16)
        self._secret = secrets.token_bytes(_MIN_SECRET_BYTES)
        self._lock = threading.Lock()
        self._last_sequence = 0
        self._closed = False
        self._revoked = False
        self._transcript = hashlib.sha256(
            canonical_json(
                {
                    "broker_transcript_v1": {
                        "capability_id": self._capability_id,
                        "launch_receipt_digest": self._launch_receipt_digest,
                        "workspace_root": grant.workspace_root,
                        "operations": sorted(grant.operations),
                    }
                }
            ).encode("utf-8")
        )

    @property
    def capability_id(self) -> str:
        return self._capability_id

    @property
    def grant(self) -> BrokerGrant:
        return self._grant

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def revoked(self) -> bool:
        with self._lock:
            return self._revoked

    def client_signer(self) -> BrokerEnvelopeSigner:
        """In-memory signer sharing this capability (tests / in-process use)."""
        return BrokerEnvelopeSigner(
            capability_id=self._capability_id,
            secret=self._secret,
            launch_receipt_digest=self._launch_receipt_digest,
        )

    @property
    def capability_id(self) -> str:
        return self._capability_id

    def reveal_secret_for_transport(self) -> bytes:
        """Host-only accessor for pipe/socketpair transport to the worker.

        Never place the returned bytes in argv, environment, logs, receipts,
        or model-visible content.
        """
        return self._secret

    def transcript_digest(self) -> str:
        """Rolling digest over every accepted, rejected, and terminal event."""
        with self._lock:
            return self._transcript.hexdigest()

    def close(self, reason: str) -> None:
        """Normal end-of-launch shutdown; every later envelope is rejected."""
        self._set_terminal("closed", reason)

    def revoke(self, reason: str) -> None:
        """Security revocation; every later envelope is rejected."""
        self._set_terminal("revoked", reason)

    def validate(self, envelope: Any) -> ValidatedBrokerRequest:
        """Fail-closed validation of one worker envelope.

        Raises :class:`BrokerEnvelopeRejected` on any defect; only a fully
        authentic, in-sequence, granted request is returned.  Every outcome
        is folded into the transcript digest.
        """
        parsed: Optional[dict[str, Any]] = None
        parse_detail = ""
        try:
            parsed = self._parse_envelope(envelope)
        except BrokerProtocolError as exc:
            parse_detail = str(exc)
        with self._lock:
            if parsed is None:
                self._reject_locked("malformed-envelope", parse_detail)
            if self._revoked:
                self._reject_locked("capability-revoked")
            if self._closed:
                self._reject_locked("capability-closed")
            if parsed["protocol_version"] != BROKER_PROTOCOL_VERSION:
                self._reject_locked("protocol-version-mismatch")
            if not hmac.compare_digest(
                parsed["capability_id"], self._capability_id
            ):
                self._reject_locked("capability-mismatch")
            expected_mac = _envelope_mac(self._secret, parsed)
            if not hmac.compare_digest(parsed["mac"], expected_mac):
                self._reject_locked("mac-invalid")
            if not hmac.compare_digest(
                parsed["launch_receipt_digest"], self._launch_receipt_digest
            ):
                self._reject_locked("launch-digest-mismatch")
            body_digest = _sha256_hex(canonical_json(parsed["body"]))
            if not hmac.compare_digest(parsed["body_digest"], body_digest):
                self._reject_locked("body-digest-mismatch")
            if parsed["sequence"] != self._last_sequence + 1:
                self._reject_locked("sequence-rejected")
            if parsed["operation"] not in self._grant.operations:
                self._reject_locked("operation-not-granted")
            self._last_sequence = parsed["sequence"]
            self._record_locked({"event": "accepted", "envelope": parsed})
            return ValidatedBrokerRequest(
                capability_id=parsed["capability_id"],
                launch_receipt_digest=parsed["launch_receipt_digest"],
                sequence=parsed["sequence"],
                operation=parsed["operation"],
                body=parsed["body"],
                body_digest=parsed["body_digest"],
                workspace_root=self._grant.workspace_root,
            )

    @staticmethod
    def _parse_envelope(envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, Mapping):
            raise BrokerProtocolError("Envelope must be a mapping.")
        if set(envelope) != _ENVELOPE_KEYS:
            raise BrokerProtocolError("Envelope keys must match the protocol exactly.")
        version = envelope["protocol_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise BrokerProtocolError("protocol_version must be an integer.")
        capability_id = envelope["capability_id"]
        if not isinstance(capability_id, str) or not _CAPABILITY_ID_RE.match(
            capability_id
        ):
            raise BrokerProtocolError(
                "capability_id must be 1-128 printable ASCII characters."
            )
        _validate_launch_digest(
            envelope["launch_receipt_digest"], "launch_receipt_digest"
        )
        sequence = envelope["sequence"]
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not 0 <= sequence <= _MAX_SEQUENCE
        ):
            raise BrokerProtocolError(
                f"sequence must be an integer in [0, {_MAX_SEQUENCE}]."
            )
        _validate_operation_name(envelope["operation"])
        _validate_launch_digest(envelope["body_digest"], "body_digest")
        mac = envelope["mac"]
        if not isinstance(mac, str) or not _HEX_DIGEST_RE.match(mac):
            raise BrokerProtocolError("mac must be a 64-char lowercase hex string.")
        _assert_json_safe(envelope["body"])
        return {key: envelope[key] for key in sorted(_ENVELOPE_KEYS)}

    def _set_terminal(self, kind: str, reason: str) -> None:
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _MAX_REASON_CHARS
        ):
            raise BrokerProtocolError(
                f"A non-empty reason of at most {_MAX_REASON_CHARS} chars is required."
            )
        with self._lock:
            if kind == "revoked":
                if not self._revoked:
                    self._revoked = True
                    self._record_locked({"event": "revoked", "reason": reason})
            else:
                if not self._closed:
                    self._closed = True
                    self._record_locked({"event": "closed", "reason": reason})

    def _reject_locked(self, reason_code: str, detail: str = "") -> NoReturn:
        self._record_locked({"event": "rejected", "reason": reason_code})
        raise BrokerEnvelopeRejected(reason_code, detail)

    def _record_locked(self, event: Mapping[str, Any]) -> None:
        self._transcript.update(canonical_json(event).encode("utf-8"))

    def __repr__(self) -> str:
        return (
            f"<SubagentBroker capability_id={self._capability_id!r} "
            f"closed={self._closed} revoked={self._revoked} secret=REDACTED>"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "SubagentBroker holds a live capability secret and must never "
            "be serialized."
        )
