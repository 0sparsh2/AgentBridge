from __future__ import annotations

import asyncio

from agentbridge import AgentSpec, arun_agent
from agentbridge_langchain.conformance import run_native_conformance


def test_langchain_native_conformance_report_passes_offline() -> None:
    report = run_native_conformance()

    assert report.passed, report.as_dict()
    assert {check.name for check in report.checks} == {
        "structured_output",
        "tools_and_normalized_events",
        "middleware_memory_retrieval",
        "human_in_the_loop",
        "observability_runtime_config",
        "native_options_escape_hatch",
    }


def test_langchain_async_run_uses_normalized_result() -> None:
    result = asyncio.run(
        arun_agent(
            AgentSpec(
                name="async_langchain_agent",
                instructions="Reply to the user.",
                model="agentbridge/offline",
            ),
            backend="langchain",
            input="hello",
        )
    )

    assert result.backend == "langchain"
    assert result.output == "offline response: hello"
    assert result.events[-1].type == "complete"
