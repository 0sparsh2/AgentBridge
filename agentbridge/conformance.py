"""Adapter conformance checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from agentbridge.registry import get_adapter, list_adapters
from agentbridge.types import AgentSpec, RunInput, ToolSpec


class ConformanceCheck(BaseModel):
    """One adapter conformance check result."""

    name: str
    passed: bool
    skipped: bool = False
    message: str = ""


class ConformanceReport(BaseModel):
    """Conformance results for one backend."""

    backend: str
    checks: list[ConformanceCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed or check.skipped for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["passed"] = self.passed
        return data


def run_conformance(
    *,
    backends: list[str] | None = None,
) -> list[ConformanceReport]:
    """Run lightweight conformance checks against selected adapters."""

    selected_backends = backends or list_adapters()
    return [_run_backend_conformance(backend) for backend in selected_backends]


def _run_backend_conformance(backend: str) -> ConformanceReport:
    adapter = get_adapter(backend)
    checks = [
        _check_capabilities(backend),
        _check_basic_run(backend),
        _check_stream(backend),
        _check_sync_tools(backend),
        _check_structured_output(backend),
    ]
    return ConformanceReport(backend=adapter.backend_name, checks=checks)


def _check_capabilities(backend: str) -> ConformanceCheck:
    def run() -> str:
        capabilities = get_adapter(backend).capabilities()
        if capabilities.backend != backend:
            raise AssertionError(f"capabilities backend {capabilities.backend!r} did not match {backend!r}")
        if "agent.instructions" not in capabilities.features:
            raise AssertionError("missing agent.instructions capability")
        return "capability metadata is present"

    return _capture("capabilities", run)


def _check_basic_run(backend: str) -> ConformanceCheck:
    def run() -> str:
        adapter = get_adapter(backend)
        spec = AgentSpec(
            name=f"{backend}_conformance_agent",
            instructions="Reply to the user.",
            model=_model_for_backend(backend),
        )
        compiled = adapter.compile(spec)
        result = adapter.run(compiled, RunInput(input="hello"))
        if result.backend != backend:
            raise AssertionError(f"result backend {result.backend!r} did not match {backend!r}")
        if result.output in (None, ""):
            raise AssertionError("result output was empty")
        return "basic run returned normalized output"

    return _capture("basic_run", run)


def _check_stream(backend: str) -> ConformanceCheck:
    def run() -> str:
        adapter = get_adapter(backend)
        spec = AgentSpec(
            name=f"{backend}_stream_agent",
            instructions="Stream a response.",
            model=_model_for_backend(backend),
        )
        compiled = adapter.compile(spec)
        events = list(adapter.stream(compiled, RunInput(input="hello")))
        if not events:
            raise AssertionError("stream returned no events")
        if events[-1].type != "complete":
            raise AssertionError("stream did not end with a complete event")
        return "stream returned normalized events"

    return _capture("stream", run)


def _check_sync_tools(backend: str) -> ConformanceCheck:
    adapter = get_adapter(backend)
    if adapter.capabilities().status("tools.sync") != "full":
        return ConformanceCheck(
            name="sync_tools",
            passed=True,
            skipped=True,
            message="backend does not advertise full sync tool support",
        )

    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    def run() -> str:
        spec = AgentSpec(
            name=f"{backend}_tool_agent",
            instructions="Use the lookup tool.",
            model=_model_for_backend(backend),
            tools=[ToolSpec.from_function(lookup_order)],
        )
        compiled = adapter.compile(spec)
        result = adapter.run(compiled, RunInput(input="A123"))
        event_types = [event.type for event in result.events]
        if "tool_call" not in event_types:
            raise AssertionError("missing tool_call event")
        if "tool_result" not in event_types:
            raise AssertionError("missing tool_result event")
        return "sync tool execution returned normalized tool events"

    return _capture("sync_tools", run)


def _check_structured_output(backend: str) -> ConformanceCheck:
    adapter = get_adapter(backend)
    if adapter.capabilities().status("structured_output") != "full":
        return ConformanceCheck(
            name="structured_output",
            passed=True,
            skipped=True,
            message="backend does not advertise full structured output support",
        )

    class Decision(BaseModel):
        eligible: bool
        reason: str

    def run() -> str:
        spec = AgentSpec(
            name=f"{backend}_structured_agent",
            instructions="Return a decision.",
            model=_model_for_backend(backend),
            output_type=Decision,
            backend_config={"custom_output_args": {"eligible": True, "reason": "ok"}},
        )
        compiled = adapter.compile(spec)
        result = adapter.run(compiled, RunInput(input="decide"))
        if not isinstance(result.output, Decision):
            raise AssertionError(f"expected Decision output, got {type(result.output).__name__}")
        return "structured output returned typed result"

    return _capture("structured_output", run)


def _capture(name: str, callback: Callable[[], str]) -> ConformanceCheck:
    try:
        message = callback()
        return ConformanceCheck(name=name, passed=True, message=message)
    except Exception as exc:
        return ConformanceCheck(name=name, passed=False, message=str(exc))


def _model_for_backend(backend: str) -> str:
    if backend == "openai_agents":
        return "agentbridge/offline"
    if backend == "strands":
        return "agentbridge/offline"
    return "test" if backend == "pydantic_ai" else "openai/gpt-5"
