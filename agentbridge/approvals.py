"""Backend-neutral helpers for human approval queues."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


ApprovalStatus = Literal["pending", "approved", "rejected"]


@dataclass(frozen=True)
class ApprovalDecision:
    """A human decision for a pending approval request."""

    approved: bool
    response: Any | None = None
    reviewer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> ApprovalStatus:
        """Return the queue status represented by this decision."""

        return "approved" if self.approved else "rejected"

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly decision record."""

        record: dict[str, Any] = {
            "approved": self.approved,
            "status": self.status,
            "response": self.response,
            "reviewer": self.reviewer,
            "metadata": dict(self.metadata),
            "decided_at": _now_iso(),
        }
        return {key: value for key, value in record.items() if value is not None}


class ApprovalQueue:
    """In-memory approval queue that adapters and apps can share.

    The queue intentionally stores plain dictionaries so records can be written by any adapter,
    serialized by an app, shown in a UI, and later turned into a resume payload.
    """

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        for record in records or []:
            self.record_approval_request(record)

    def record_approval_request(self, record: dict[str, Any]) -> dict[str, Any]:
        """Add or replace a pending approval request."""

        normalized = _normalize_record(record)
        self._records[normalized["id"]] = normalized
        return deepcopy(normalized)

    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        """Alias used by adapter store integrations."""

        return self.record_approval_request(record)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        """Alias used by list-like adapter store integrations."""

        return self.record_approval_request(record)

    def get(self, request_id: str) -> dict[str, Any]:
        """Return a stored approval request by id."""

        try:
            return deepcopy(self._records[request_id])
        except KeyError as exc:
            raise KeyError(f"Unknown approval request: {request_id}") from exc

    def list(self, *, status: ApprovalStatus | None = None, backend: str | None = None) -> list[dict[str, Any]]:
        """List approval records, optionally filtered by status or backend."""

        records = list(self._records.values())
        if status is not None:
            records = [record for record in records if record.get("status") == status]
        if backend is not None:
            records = [record for record in records if record.get("backend") == backend]
        return [deepcopy(record) for record in records]

    def pending(self, *, backend: str | None = None) -> list[dict[str, Any]]:
        """Return pending approval requests."""

        return self.list(status="pending", backend=backend)

    def decide(
        self,
        request_id: str,
        decision: ApprovalDecision,
    ) -> dict[str, Any]:
        """Apply a decision and return a backend-neutral resume payload."""

        record = self.get(request_id)
        if record.get("status") != "pending":
            raise ValueError(f"Approval request {request_id!r} is already {record.get('status')}.")

        decision_record = decision.to_record()
        record["status"] = decision.status
        record["decision"] = decision_record
        self._records[request_id] = record
        return self.resume_payload(request_id)

    def approve(
        self,
        request_id: str,
        *,
        response: Any | None = None,
        reviewer: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Approve a request and return a resume payload."""

        return self.decide(
            request_id,
            ApprovalDecision(
                approved=True,
                response=response,
                reviewer=reviewer,
                metadata=metadata or {},
            ),
        )

    def reject(
        self,
        request_id: str,
        *,
        response: Any | None = None,
        reviewer: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject a request and return a resume payload."""

        return self.decide(
            request_id,
            ApprovalDecision(
                approved=False,
                response=response,
                reviewer=reviewer,
                metadata=metadata or {},
            ),
        )

    def resume_payload(self, request_id: str) -> dict[str, Any]:
        """Build the portable payload an app can pass to a backend-specific resume path."""

        record = self.get(request_id)
        return {
            "id": record["id"],
            "backend": record.get("backend"),
            "status": record.get("status"),
            "decision": deepcopy(record.get("decision")),
            "interruption": deepcopy(record.get("interruption")),
            "state": deepcopy(record.get("state")),
            "state_type": record.get("state_type"),
            "last_response_id": record.get("last_response_id"),
            "last_agent": deepcopy(record.get("last_agent")),
            "metadata": deepcopy(record.get("metadata", {})),
        }


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    request_id = normalized.get("id")
    if request_id is None:
        raise ValueError("Approval records require an 'id'.")
    normalized["id"] = str(request_id)
    normalized.setdefault("status", "pending")
    normalized.setdefault("created_at", _now_iso())
    normalized.setdefault("metadata", {})
    return normalized


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
