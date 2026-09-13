from __future__ import annotations

import pytest

from agentbridge import ApprovalDecision, ApprovalQueue


def test_approval_queue_records_and_filters_pending_requests() -> None:
    queue = ApprovalQueue()

    stored = queue.record_approval_request(
        {
            "id": "approval-1",
            "backend": "openai_agents",
            "interruption": {"tool_name": "issue_refund"},
            "state": {"id": "state-1"},
            "last_response_id": "resp-1",
            "last_agent": {"name": "billing_agent"},
        }
    )

    assert stored["status"] == "pending"
    assert stored["created_at"]
    assert queue.pending(backend="openai_agents")[0]["id"] == "approval-1"
    assert queue.pending(backend="strands") == []


def test_approval_queue_decision_returns_resume_payload() -> None:
    queue = ApprovalQueue(
        [
            {
                "id": "approval-1",
                "backend": "openai_agents",
                "interruption": {"tool_name": "issue_refund"},
                "state": {"id": "state-1"},
                "state_type": "RunState",
                "last_response_id": "resp-1",
                "last_agent": {"name": "billing_agent"},
            }
        ]
    )

    payload = queue.approve(
        "approval-1",
        response={"approved": True},
        reviewer="support-lead",
        metadata={"ticket": "T-1"},
    )

    assert payload["id"] == "approval-1"
    assert payload["backend"] == "openai_agents"
    assert payload["status"] == "approved"
    assert payload["decision"]["approved"] is True
    assert payload["decision"]["response"] == {"approved": True}
    assert payload["decision"]["reviewer"] == "support-lead"
    assert payload["decision"]["metadata"] == {"ticket": "T-1"}
    assert payload["state"] == {"id": "state-1"}
    assert payload["last_response_id"] == "resp-1"
    assert queue.pending() == []


def test_approval_queue_rejects_unknown_or_duplicate_decisions() -> None:
    queue = ApprovalQueue([{"id": "approval-1"}])

    with pytest.raises(KeyError, match="Unknown approval request"):
        queue.approve("missing")

    queue.reject("approval-1", response="not allowed")

    with pytest.raises(ValueError, match="already rejected"):
        queue.approve("approval-1")


def test_approval_decision_serializes_status() -> None:
    approved = ApprovalDecision(approved=True).to_record()
    rejected = ApprovalDecision(approved=False).to_record()

    assert approved["status"] == "approved"
    assert rejected["status"] == "rejected"
