from agentbridge.observability import observability_metadata


def test_observability_metadata_correlates_langsmith_langfuse_and_session():
    metadata = observability_metadata(
        {
            "metadata": {"service": "refunds"},
            "observability": {
                "langsmith": {"enabled": True, "project_name": "support", "trace_id": "ls-1"},
                "langfuse": {"enabled": True, "user_id": "user-1", "trace_id": "lf-1"},
            },
        },
        metadata={"request_id": "req-1"},
        session_id="session-1",
    )
    assert metadata == {
        "service": "refunds",
        "request_id": "req-1",
        "session_id": "session-1",
        "langsmith_project": "support",
        "langsmith_trace_id": "ls-1",
        "langfuse_user_id": "user-1",
        "langfuse_trace_id": "lf-1",
        "langfuse_session_id": "session-1",
    }
