"""AgentBridge adapter plugin for Google ADK."""

from __future__ import annotations

import warnings
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
            "model": _model_for_spec(sdk, spec),
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


def _model_for_spec(sdk: dict[str, Any], spec: AgentSpec) -> Any:
    if spec.model == "agentbridge/offline":
        base_llm = import_module("google.adk.models.base_llm")
        llm_response = import_module("google.adk.models.llm_response")

        class AgentBridgeOfflineModel(_AgentBridgeOfflineModelBase, base_llm.BaseLlm):
            pass

        return AgentBridgeOfflineModel(
            model="agentbridge/offline",
            response_type=llm_response.LlmResponse,
            types=sdk["types"],
        )
    return spec.model


class _AgentBridgeOfflineModelBase:
    """No-network Google ADK model used by conformance tests."""

    response_type: Any
    types: Any

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        tool_response = _latest_function_response(llm_request.contents)
        if tool_response is not None:
            yield self._text_response(f"offline tool result: {tool_response}")
            return
        if llm_request.tools_dict:
            tool_name, tool = next(iter(llm_request.tools_dict.items()))
            argument_name = _first_tool_argument_name(tool)
            yield self.response_type(
                content=self.types.Content(
                    role="model",
                    parts=[
                        self.types.Part.from_function_call(
                            name=tool_name,
                            args={argument_name: _last_user_text(llm_request.contents)},
                        )
                    ],
                ),
                usage_metadata=self._usage_metadata(),
            )
            return
        yield self._text_response(f"offline response: {_last_user_text(llm_request.contents)}")

    def _text_response(self, text: str) -> Any:
        return self.response_type(
            content=self.types.Content(
                role="model",
                parts=[self.types.Part.from_text(text=text)],
            ),
            usage_metadata=self._usage_metadata(),
        )

    def _usage_metadata(self) -> Any:
        return self.types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
        )


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


def _first_tool_argument_name(tool: Any) -> str:
    declaration = None
    if hasattr(tool, "_get_declaration"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            declaration = tool._get_declaration()
    schema = getattr(declaration, "parameters_json_schema", None) if declaration else None
    if isinstance(schema, dict):
        required = schema.get("required") or []
        if required:
            return str(required[0])
        properties = schema.get("properties") or {}
        if properties:
            return str(next(iter(properties)))
    return "input"


def _latest_function_response(contents: list[Any]) -> Any | None:
    for content in reversed(contents or []):
        for part in reversed(getattr(content, "parts", []) or []):
            function_response = getattr(part, "function_response", None)
            if function_response is None:
                continue
            response = getattr(function_response, "response", None)
            if isinstance(response, dict):
                if "result" in response:
                    return response["result"]
                if "content" in response:
                    return response["content"]
            return response
    return None


def _last_user_text(contents: list[Any]) -> str:
    for content in reversed(contents or []):
        if getattr(content, "role", None) != "user":
            continue
        chunks: list[str] = []
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
        if chunks:
            return "".join(chunks)
    return ""


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
    function_call = _event_function_call(event)
    if function_call:
        payload["function_call"] = function_call
    function_response = _event_function_response(event)
    if function_response:
        payload["function_response"] = function_response
    keys = {str(key).lower() for key in payload}
    if function_response:
        event_type = "tool_result"
    elif function_call:
        event_type = "tool_call"
    elif text:
        event_type = "message"
    elif any("tool" in key and ("result" in key or "response" in key) for key in keys):
        event_type = "tool_result"
    elif any("tool" in key or "function_call" in key for key in keys):
        event_type = "tool_call"
    elif any("error" in key or "exception" in key for key in keys):
        event_type = "error"
    else:
        event_type = "message"
    payload.setdefault("native_type", type(event).__name__)
    return AgentEvent(type=event_type, backend=backend, data=payload)


def _event_function_call(event: Any) -> dict[str, Any] | None:
    for part in _event_parts(event):
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        return _payload(function_call)
    return None


def _event_function_response(event: Any) -> dict[str, Any] | None:
    for part in _event_parts(event):
        function_response = getattr(part, "function_response", None)
        if function_response is None:
            continue
        return _payload(function_response)
    return None


def _event_parts(event: Any) -> list[Any]:
    content = getattr(event, "content", None)
    if content is None and isinstance(event, dict):
        content = event.get("content")
    parts = getattr(content, "parts", None)
    if parts is None and isinstance(content, dict):
        parts = content.get("parts")
    return list(parts or [])


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}
