"""AgentBridge adapter plugin for Google ADK."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult


@dataclass(frozen=True)
class CompiledGoogleADKAgent:
    """Compiled Google ADK agent and runner objects."""

    spec: AgentSpec
    native_agent: Any
    native_tools: list[Any]
    runner: Any
    types: Any
    config: dict[str, Any]


class Adapter(BackendAdapter):
    """Adapter entry point discovered by AgentBridge."""

    backend_name = "google_adk"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.instructions": "full",
                "agent.model": "partial",
                "tools.sync": "full",
                "state.session": "extension",
                "state.memory": "extension",
                "workflow.delegation": "extension",
                "deployment.serverless": "extension",
                "evals": "extension",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec instructions to ADK Agent instruction.",
                "agent.model": "Passes model strings through to ADK; provider compatibility is ADK/model dependent.",
                "tools.sync": "Maps ToolSpec callables to ADK FunctionTool instances.",
                "state.session": "Planned through GoogleADKExtension session_service and RunInput.session_id.",
                "state.memory": "Planned through GoogleADKExtension memory_service.",
                "deployment.serverless": "Planned as extension-level deployment metadata.",
                "streaming.events": "Normalizes Runner.run events best-effort.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledGoogleADKAgent:
        """Translate AgentSpec into Google ADK agent and runner objects."""

        sdk = _load_google_adk()
        config = dict(spec.backend_config.get(self.backend_name, {}))
        native_tools = [sdk["FunctionTool"](tool.handler) for tool in spec.tools]
        agent_kwargs: dict[str, Any] = {
            "name": spec.name,
            "model": spec.model,
            "instruction": spec.instructions,
            "tools": native_tools,
        }
        if spec.output_schema is not None:
            agent_kwargs["output_schema"] = spec.output_schema
        if config.get("sub_agents"):
            agent_kwargs["sub_agents"] = config["sub_agents"]
        native_agent = sdk["Agent"](**agent_kwargs)
        session_service = config.get("session_service")
        if session_service in (None, "memory", "in_memory"):
            session_service = sdk["InMemorySessionService"]()
        runner = sdk["Runner"](
            app_name=config.get("app_name") or spec.name,
            agent=native_agent,
            session_service=session_service,
            memory_service=config.get("memory_service"),
            auto_create_session=True,
        )
        return CompiledGoogleADKAgent(
            spec=spec,
            native_agent=native_agent,
            native_tools=native_tools,
            runner=runner,
            types=sdk["types"],
            config=config,
        )

    def run(self, compiled: Any, run_input: RunInput) -> RunResult:
        """Run the compiled Google ADK agent and return a normalized result."""

        compiled_agent = _ensure_compiled(compiled)
        events = list(_run_events(compiled_agent, run_input))
        output = _final_output(events)
        normalized_events = [
            _normalize_event(event, backend=self.backend_name) for event in events
        ]
        normalized_events.append(
            AgentEvent(
                type="complete",
                backend=self.backend_name,
                data={"output": output},
            )
        )
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=normalized_events,
            metadata={"agent": compiled_agent.spec.name},
            raw=events,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        events: list[Any] = []
        for event in _run_events(compiled_agent, run_input):
            events.append(event)
            yield _normalize_event(event, backend=self.backend_name)
        yield AgentEvent(
            type="complete",
            backend=self.backend_name,
            data={"output": _final_output(events)},
        )


def _load_google_adk() -> dict[str, Any]:
    try:
        agents = import_module("google.adk.agents")
        runners = import_module("google.adk.runners")
        sessions = import_module("google.adk.sessions")
        function_tool = import_module("google.adk.tools.function_tool")
        genai_types = import_module("google.genai.types")
    except ImportError as exc:  # pragma: no cover - message assertion path.
        raise ImportError(
            "Google ADK is not installed or one of its optional runtime dependencies is missing. "
            "Install this plugin with `google-adk>=2.9,<3`. Native ADK and Strands installs may "
            "need isolated environments because they currently require different OpenTelemetry ranges."
        ) from exc
    return {
        "Agent": agents.Agent,
        "Runner": runners.Runner,
        "InMemorySessionService": sessions.InMemorySessionService,
        "FunctionTool": function_tool.FunctionTool,
        "types": genai_types,
    }


def _ensure_compiled(compiled: Any) -> CompiledGoogleADKAgent:
    if not isinstance(compiled, CompiledGoogleADKAgent):
        raise TypeError("Google ADK adapter expected CompiledGoogleADKAgent from compile().")
    return compiled


def _run_events(compiled: CompiledGoogleADKAgent, run_input: RunInput) -> Iterator[Any]:
    types = compiled.types
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=run_input.input)],
    )
    yield from compiled.runner.run(
        user_id=run_input.metadata.get("user_id", "agentbridge"),
        session_id=run_input.session_id or "default",
        new_message=content,
        state_delta=run_input.context or None,
    )


def _final_output(events: list[Any]) -> Any:
    for event in reversed(events):
        text = _event_text(event)
        if text:
            return text
    return ""


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    if content is None and isinstance(event, dict):
        content = event.get("content")
    parts = getattr(content, "parts", None)
    if parts is None and isinstance(content, dict):
        parts = content.get("parts")
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            chunks.append(str(text))
    return "".join(chunks)


def _normalize_event(event: Any, *, backend: str) -> AgentEvent:
    payload = _payload(event)
    text = _event_text(event)
    if text:
        payload["content"] = text
    keys = {str(key).lower() for key in payload}
    if any("tool" in key and ("result" in key or "response" in key) for key in keys):
        event_type = "tool_result"
    elif any("tool" in key or "function_call" in key for key in keys):
        event_type = "tool_call"
    elif any("error" in key or "exception" in key for key in keys):
        event_type = "error"
    else:
        event_type = "message"
    payload.setdefault("native_type", type(event).__name__)
    return AgentEvent(type=event_type, backend=backend, data=payload)


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}
