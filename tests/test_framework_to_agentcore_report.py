from examples.framework_to_agentcore_report import build_framework_to_agentcore_report


def test_framework_to_agentcore_report_is_redacted_and_comparable():
    report = build_framework_to_agentcore_report(
        agent_name="refund",
        framework_results={"langchain": {"answer": "yes"}, "langgraph": None},
        agentcore_config={
            "runtime_arn": "arn:runtime",
            "memory_id": "memory-1",
            "gateway_url": "https://gateway.example",
            "secret": "do-not-copy",
        },
    )
    assert report["frameworks"]["langchain"]["output_present"] is True
    assert report["frameworks"]["langgraph"]["output_present"] is False
    assert report["agentcore"]["memory_id_present"] is True
    assert "secret" not in str(report)
