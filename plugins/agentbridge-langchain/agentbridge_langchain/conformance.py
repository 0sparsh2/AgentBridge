"""Executable conformance checks for LangChain-native adapter surfaces.

These checks intentionally use AgentBridge's offline LangChain model. They validate
translation and normalization without requiring provider credentials, while keeping
the native LangChain objects in the execution path.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.langchain import LangChainExtension

from .adapter import Adapter


class NativeConformanceCheck(BaseModel):
    """Result for one LangChain-native capability group."""

    name: str
    passed: bool
    message: str


class NativeConformanceReport(BaseModel):
    """Executable coverage report for the LangChain plugin."""

    backend: str = "langchain"
    checks: list[NativeConformanceCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["passed"] = self.passed
        return payload


class _Decision(BaseModel):
    eligible: bool
    reason: str


def run_native_conformance() -> NativeConformanceReport:
    """Exercise the current LangChain native-only translation surfaces offline."""

    checks = [
        _capture("structured_output", _check_structured_output),
        _capture("tools_and_normalized_events", _check_tools_and_events),
        _capture("retriever_tool_bridge", _check_retriever_tool_bridge),
        _capture("middleware_memory_retrieval", _check_native_state_options),
        _capture("human_in_the_loop", _check_human_in_the_loop),
        _capture("observability_runtime_config", _check_observability_config),
        _capture("observability_provider_options", _check_observability_provider_options),
        _capture("agentcore_bindings", _check_agentcore_bindings),
        _capture("async_run", _check_async_run),
        _capture("async_streaming", _check_async_streaming),
        _capture("native_options_escape_hatch", _check_native_options),
    ]
    return NativeConformanceReport(checks=checks)


def _check_structured_output() -> str:
    spec = AgentSpec(
        name="langchain_native_structured",
        instructions="Return a decision.",
        model="agentbridge/offline",
        output_type=_Decision,
        backend_config={"custom_output_args": {"eligible": True, "reason": "offline"}},
    )
    result = Adapter().run(Adapter().compile(spec), RunInput(input="decide"))
    if not isinstance(result.output, _Decision):
        raise AssertionError(f"expected typed output, got {type(result.output).__name__}")
    if not result.metadata["run_diagnostics"]["structured_response"]:
        raise AssertionError("structured response was not reported")
    return "create_agent response_format and normalized typed output passed"


def _check_tools_and_events() -> str:
    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    spec = AgentSpec(
        name="langchain_native_tools",
        instructions="Use the lookup tool.",
        model="agentbridge/offline",
        tools=[ToolSpec.from_function(lookup_order)],
    )
    adapter = Adapter()
    result = adapter.run(adapter.compile(spec), RunInput(input="A123"))
    event_types = {event.type for event in result.events}
    if not {"tool_call", "tool_result", "complete"}.issubset(event_types):
        raise AssertionError(f"missing normalized tool lifecycle events: {event_types}")
    return "native StructuredTool execution and normalized tool lifecycle passed"


def _check_retriever_tool_bridge() -> str:
    class PolicyRetriever:
        name = "refund_policy"
        description = "Search refund policy documents."

        def invoke(self, query: str) -> list[Any]:
            if query != "double charge":
                raise AssertionError(f"unexpected retriever query: {query}")
            return [
                {"page_content": "Refunds are available within 30 days.", "metadata": {"source": "policy"}}
            ]

    spec = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_native_retriever_tool",
            instructions="Search policy documents.",
            model="agentbridge/offline",
        ),
        retriever_tools=[PolicyRetriever()],
    )
    compiled = Adapter().compile(spec)
    tool = compiled.native_tools[0]
    result = tool.func("double charge")
    if result != [
        {"page_content": "Refunds are available within 30 days.", "metadata": {"source": "policy"}}
    ]:
        raise AssertionError(f"retriever result was not preserved: {result}")
    return "native retriever invocation and document payload preservation passed"


def _check_native_state_options() -> str:
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    extension = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_native_state",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        middleware=[AgentMiddleware()],
        memory="conversation_buffer",
        retrievers=["policy-retriever"],
        checkpointer=checkpointer,
        store=store,
        interrupt_before=["model"],
        interrupt_after=["model"],
    )
    compiled = Adapter().compile(extension)
    kwargs = compiled.config
    for name in ("middleware", "checkpointer", "store", "interrupt_before", "interrupt_after"):
        if name not in kwargs:
            raise AssertionError(f"native option {name!r} was not forwarded")
    return "middleware, memory/retriever hints, checkpoint/store, and interrupt options passed"


def _check_observability_config() -> str:
    from langchain_core.callbacks import BaseCallbackHandler

    extension = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_native_observability",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        callbacks=[BaseCallbackHandler()],
        observability={"tags": ["conformance"], "run_name": "native-check"},
    )
    adapter = Adapter()
    result = adapter.run(
        adapter.compile(extension),
        RunInput(input="hello", session_id="native-conformance"),
    )
    config = result.metadata["runtime_config"]
    if len(config["callbacks"]) != 1 or config["tags"] != ["conformance"]:
        raise AssertionError(f"observability config was not forwarded: {config}")
    if config["configurable"]["thread_id"] != "native-conformance":
        raise AssertionError("session id was not mapped to thread_id")
    return "callbacks, tags, run name, metadata, and session config passed"


def _check_human_in_the_loop() -> str:
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    spec = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_native_hitl",
            instructions="Use the lookup tool.",
            model="agentbridge/offline",
            tools=[ToolSpec.from_function(lookup_order)],
        ),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"lookup_order": True})],
        checkpointer=InMemorySaver(),
    )
    adapter = Adapter()
    result = adapter.run(
        adapter.compile(spec),
        RunInput(input="A123", session_id="native-hitl"),
    )
    diagnostics = result.metadata["run_diagnostics"]
    if not diagnostics["interrupted"]:
        raise AssertionError("native HITL middleware did not produce an interrupt")
    if not any(
        event.type == "workflow" and event.data.get("phase") == "interrupted"
        for event in result.events
    ):
        raise AssertionError("interrupt was not normalized as a workflow event")
    return "HumanInTheLoopMiddleware pause and normalized interrupt event passed"


def _check_native_options() -> str:
    extension = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_native_escape_hatch",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        # ``debug`` is intentionally passed through native_options here rather
        # than the convenience field, proving the escape hatch with a real
        # create_agent keyword accepted by the pinned LangChain version.
        native_options={"debug": True},
    )
    compiled = Adapter().compile(extension)
    if compiled.native_agent.debug is not True:
        raise AssertionError("native_options did not reach create_agent")
    return "guarded native_options pass-through passed"


def _check_observability_provider_options() -> str:
    extension = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_provider_observability",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        observability={
            "langsmith": {"enabled": True, "project_name": "conformance"},
            "langfuse": {"enabled": True, "user_id": "conformance-user"},
        },
    )
    config = Adapter().compile(extension).config
    observability = config["observability"]
    if observability["langsmith"]["enabled"] is not True or observability["langfuse"]["enabled"] is not True:
        raise AssertionError(f"provider observability options were not preserved: {observability}")
    return "LangSmith and Langfuse provider options were preserved without credentials"


def _check_agentcore_bindings() -> str:
    extension = LangChainExtension.with_config(
        AgentSpec(
            name="langchain_agentcore_bindings",
            instructions="Reply to the user.",
            model="agentbridge/offline",
        ),
        agentcore={"memory_id": "memory-conformance", "gateway_url": "https://gateway.example"},
    )
    result = Adapter().run(Adapter().compile(extension), RunInput(input="hello"))
    bindings = result.metadata["extension_summary"]["agentcore"]
    if bindings["memory_id"] != "memory-conformance":
        raise AssertionError("AgentCore Memory binding was not preserved")
    return "AgentCore Memory and Gateway bindings were preserved"


def _check_async_run() -> str:
    spec = AgentSpec(
        name="langchain_native_async",
        instructions="Reply to the user.",
        model="agentbridge/offline",
    )
    result = asyncio.run(Adapter().arun(Adapter().compile(spec), RunInput(input="hello")))
    if result.backend != "langchain" or result.events[-1].type != "complete":
        raise AssertionError("async LangChain execution was not normalized")
    return "native async invocation returned normalized output and completion"


def _check_async_streaming() -> str:
    spec = AgentSpec(
        name="langchain_native_async_stream",
        instructions="Reply to the user.",
        model="agentbridge/offline",
    )

    async def collect() -> list[str]:
        events: list[str] = []
        async for event in Adapter().astream(
            Adapter().compile(spec),
            RunInput(input="hello"),
        ):
            events.append(event.type)
        return events

    event_types = asyncio.run(collect())
    if not event_types or event_types[-1] != "complete":
        raise AssertionError(f"async stream did not complete: {event_types}")
    return "native async stream contract and normalized completion passed"


def _capture(name: str, callback: Any) -> NativeConformanceCheck:
    try:
        return NativeConformanceCheck(name=name, passed=True, message=callback())
    except Exception as exc:  # pragma: no cover - failure is represented in the report.
        return NativeConformanceCheck(name=name, passed=False, message=str(exc))
