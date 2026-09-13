from __future__ import annotations

import os

import pytest

from agentbridge import AgentSpec, RunInput
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge_langchain.adapter import Adapter


pytestmark = pytest.mark.skipif(
    os.getenv("AGENTBRIDGE_LANGSMITH_SMOKE") != "1"
    or not os.getenv("LANGSMITH_API_KEY"),
    reason="Set AGENTBRIDGE_LANGSMITH_SMOKE=1 and LANGSMITH_API_KEY to run LangSmith smoke tests.",
)


def test_langchain_adapter_runs_inside_langsmith_tracing_context() -> None:
    from langsmith.run_helpers import tracing_context

    adapter = Adapter()
    agent = LangChainExtension.with_config(
        AgentSpec(
            name="langsmith_smoke_agent",
            instructions="Reply offline while LangSmith tracing is enabled.",
            model="agentbridge/offline",
        ),
        metadata={"agentbridge_smoke": "langsmith"},
    )

    with tracing_context(
        project_name=os.getenv("LANGSMITH_PROJECT", "agentbridge-smoke"),
        tags=["agentbridge", "langchain", "smoke"],
        metadata={"backend": "langchain"},
        enabled=True,
    ):
        compiled = adapter.compile(agent)
        result = adapter.run(
            compiled,
            RunInput(
                input="hello",
                metadata={"request_id": "langsmith-smoke"},
                session_id="langsmith-smoke-session",
            ),
        )

    assert result.output == "offline response: hello"
    assert result.metadata["runtime_config"]["metadata"] == {
        "agentbridge_smoke": "langsmith",
        "request_id": "langsmith-smoke",
        "session_id": "langsmith-smoke-session",
    }
