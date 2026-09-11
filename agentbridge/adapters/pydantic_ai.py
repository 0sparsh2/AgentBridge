"""Pydantic AI adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentbridge.adapters.base import BackendAdapter
from agentbridge.errors import MissingDependencyError
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class PydanticAICompiledAgent:
    spec: AgentSpec
    agent: Any


class PydanticAIAdapter(BackendAdapter):
    """Adapter for Pydantic AI."""

    backend_name = "pydantic_ai"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "full",
                "tools.sync": "full",
                "tools.async": "partial",
                "structured_output": "full",
                "workflow.graph": "unsupported",
                "workflow.roles_tasks": "unsupported",
                "state.session": "partial",
                "streaming.events": "partial",
                "human_approval": "unsupported",
                "observability.raw": "full",
                "agui.events": "partial",
            },
            notes={
                "structured_output": "Typed outputs are a natural Pydantic AI strength.",
                "tools.async": "Native support exists, but AgentBridge ToolSpec is sync-first today.",
                "streaming.events": "Current adapter normalizes final events; richer streaming is planned.",
            },
        )

    def compile(self, spec: AgentSpec) -> PydanticAICompiledAgent:
        try:
            from pydantic_ai import Agent
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "pydantic-ai-slim", "pydantic-ai") from exc

        model = self._resolve_model(spec)
        agent = Agent(
            model=model,
            output_type=spec.output_type or str,
            instructions=spec.instructions,
            name=spec.name,
            defer_model_check=True,
        )
        for tool in spec.tools:
            agent.tool_plain(tool.handler, name=tool.name, description=tool.description)
        return PydanticAICompiledAgent(spec=spec, agent=agent)

    def run(self, compiled: PydanticAICompiledAgent, run_input: RunInput) -> RunResult:
        raw = compiled.agent.run_sync(run_input.input)
        output = getattr(raw, "output", None)
        if output is None:
            output = getattr(raw, "data", raw)
        events = self._events_from_raw(raw, output)
        usage = self._usage_from_raw(raw)
        return RunResult(output=output, backend=self.backend_name, events=events, usage=usage, raw=raw)

    def _resolve_model(self, spec: AgentSpec) -> Any:
        if spec.model != "test":
            return _to_pydantic_ai_model_name(spec.model)

        try:
            from pydantic_ai.models.test import TestModel
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise MissingDependencyError(self.backend_name, "pydantic-ai-slim", "pydantic-ai") from exc

        return TestModel(
            custom_output_text=spec.backend_config.get("custom_output_text"),
            custom_output_args=spec.backend_config.get("custom_output_args"),
        )

    def _events_from_raw(self, raw: Any, output: Any) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        for message in raw.all_messages():
            for part in getattr(message, "parts", []):
                part_type = type(part).__name__
                if part_type == "ToolCallPart":
                    events.append(
                        AgentEvent(
                            type="tool_call",
                            backend=self.backend_name,
                            data={
                                "name": part.tool_name,
                                "arguments": part.args,
                                "tool_call_id": part.tool_call_id,
                            },
                        )
                    )
                elif part_type == "ToolReturnPart":
                    events.append(
                        AgentEvent(
                            type="tool_result",
                            backend=self.backend_name,
                            data={
                                "name": part.tool_name,
                                "result": part.content,
                                "tool_call_id": part.tool_call_id,
                            },
                        )
                    )

        events.append(
            AgentEvent(
                type="message",
                backend=self.backend_name,
                data={"role": "assistant", "content": output},
            )
        )
        events.append(AgentEvent(type="complete", backend=self.backend_name, data={"output": output}))
        return events

    def _usage_from_raw(self, raw: Any) -> dict[str, Any]:
        usage_attr = getattr(raw, "usage", None)
        usage = usage_attr() if callable(usage_attr) else usage_attr
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        return dict(usage) if isinstance(usage, dict) else {}


def _to_pydantic_ai_model_name(model: str) -> str:
    """Translate LiteLLM-style provider/model names to Pydantic AI names."""

    if ":" in model:
        return model
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return f"{provider}:{model_name}"
    return model
