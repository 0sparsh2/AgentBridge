"""AgentBridge adapter plugin for Google ADK."""

from __future__ import annotations

import warnings
import json
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
                "structured_output": "full",
                "state.session": "extension",
                "state.memory": "extension",
                "workflow.delegation": "extension",
                "deployment.serverless": "extension",
                "evals": "extension",
                "observability.diagnostics": "full",
                "observability.raw": "full",
                "streaming.events": "partial",
            },
            notes={
                "agent.instructions": "Maps AgentSpec instructions to ADK Agent instruction.",
                "agent.model": "Passes model strings through to ADK; provider compatibility is ADK/model dependent.",
                "tools.sync": "Maps ToolSpec callables to ADK FunctionTool instances.",
                "structured_output": "Passes AgentSpec.output_schema to ADK and validates JSON text into AgentSpec.output_type when available.",
                "state.session": "Forwards native session_service/session ids and records run session metadata.",
                "state.memory": "Forwards native memory_service and records memory metadata.",
                "workflow.delegation": "Forwards native sub_agents and transfer controls when supplied.",
                "deployment.serverless": "Records deployment_target metadata for ADK deployment paths.",
                "evals": "Records eval metadata; native eval execution remains extension-level.",
                "observability.diagnostics": "Normalizes event history, session/service bindings, and transfer-to-agent targets into run_diagnostics metadata.",
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
        _copy_native_options(
            config,
            agent_kwargs,
            (
                "description",
                "global_instruction",
                "static_instruction",
                "input_schema",
                "state_schema",
                "generate_content_config",
                "mode",
                "parallel_worker",
                "disallow_transfer_to_parent",
                "disallow_transfer_to_peers",
                "include_contents",
                "output_key",
                "planner",
                "code_executor",
                "retry_config",
                "timeout",
                "rerun_on_resume",
                "wait_for_output",
                "before_agent_callback",
                "after_agent_callback",
                "before_model_callback",
                "after_model_callback",
                "on_model_error_callback",
                "before_tool_callback",
                "after_tool_callback",
                "on_tool_error_callback",
                "sub_agents",
            ),
        )
        native_agent = sdk["Agent"](**agent_kwargs)
        session_service = config.get("session_service")
        if session_service in (None, "memory", "in_memory"):
            session_service = sdk["InMemorySessionService"]()
        runner_kwargs: dict[str, Any] = {
            "app_name": config.get("app_name") or spec.name,
            "agent": native_agent,
            "session_service": session_service,
            "memory_service": config.get("memory_service"),
            "auto_create_session": config.get("auto_create_session", True),
        }
        _copy_native_options(
            config,
            runner_kwargs,
            (
                "artifact_service",
                "credential_service",
                "plugin_close_timeout",
            ),
        )
        if config.get("runner_plugins"):
            runner_kwargs["plugins"] = config["runner_plugins"]
        runner = sdk["Runner"](
            **runner_kwargs,
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
        run_kwargs = _run_kwargs(compiled_agent, run_input)
        events = list(_run_events(compiled_agent, run_kwargs))
        output = _final_output(events, compiled_agent.spec)
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
            metadata={
                "agent": compiled_agent.spec.name,
                "run_kwargs": _safe_summary(run_kwargs),
                "extension_config": _safe_summary(compiled_agent.config),
                "extension_summary": _extension_summary(compiled_agent.config, run_kwargs),
                "native_agent_type": type(compiled_agent.native_agent).__name__,
                "native_runner_type": type(compiled_agent.runner).__name__,
                "run_diagnostics": _run_diagnostics(compiled_agent, events, run_kwargs),
            },
            raw=events,
        )

    def stream(self, compiled: Any, run_input: RunInput) -> Iterator[AgentEvent]:
        compiled_agent = _ensure_compiled(compiled)
        events: list[Any] = []
        for event in _run_events(compiled_agent, _run_kwargs(compiled_agent, run_input)):
            events.append(event)
            yield _normalize_event(event, backend=self.backend_name)
        yield AgentEvent(
            type="complete",
            backend=self.backend_name,
            data={"output": _final_output(events, compiled_agent.spec)},
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


def _copy_native_options(
    config: dict[str, Any],
    kwargs: dict[str, Any],
    option_names: tuple[str, ...],
) -> None:
    for option_name in option_names:
        if option_name not in config:
            continue
        value = config[option_name]
        if value is None:
            continue
        if option_name in {
            "parallel_worker",
            "disallow_transfer_to_parent",
            "disallow_transfer_to_peers",
            "rerun_on_resume",
            "wait_for_output",
            "auto_create_session",
        }:
            kwargs[option_name] = bool(value)
        else:
            kwargs[option_name] = value


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
        response_schema = _response_schema(llm_request)
        if response_schema:
            yield self._text_response(json.dumps(_value_for_schema(response_schema, llm_request.contents)))
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


def _run_kwargs(compiled: CompiledGoogleADKAgent, run_input: RunInput) -> dict[str, Any]:
    types = compiled.types
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=run_input.input)],
    )
    return {
        "user_id": run_input.metadata.get("user_id", "agentbridge"),
        "session_id": run_input.session_id or "default",
        "new_message": content,
        "state_delta": run_input.context or None,
    }


def _run_events(compiled: CompiledGoogleADKAgent, run_kwargs: dict[str, Any]) -> Iterator[Any]:
    yield from compiled.runner.run(**run_kwargs)


def _extension_summary(config: dict[str, Any], run_kwargs: dict[str, Any]) -> dict[str, Any]:
    native_agent_options = [
        "description",
        "global_instruction",
        "static_instruction",
        "input_schema",
        "state_schema",
        "generate_content_config",
        "mode",
        "parallel_worker",
        "disallow_transfer_to_parent",
        "disallow_transfer_to_peers",
        "include_contents",
        "output_key",
        "planner",
        "code_executor",
        "retry_config",
        "timeout",
        "rerun_on_resume",
        "wait_for_output",
        "before_agent_callback",
        "after_agent_callback",
        "before_model_callback",
        "after_model_callback",
        "on_model_error_callback",
        "before_tool_callback",
        "after_tool_callback",
        "on_tool_error_callback",
        "sub_agents",
    ]
    native_runner_options = [
        "session_service",
        "memory_service",
        "artifact_service",
        "credential_service",
        "runner_plugins",
        "plugin_close_timeout",
        "auto_create_session",
    ]
    return {
        "app_name": config.get("app_name"),
        "session_service": _safe_summary(config.get("session_service", "in_memory")),
        "memory_service": _safe_summary(config.get("memory_service")),
        "artifact_service": _safe_summary(config.get("artifact_service")),
        "credential_service": _safe_summary(config.get("credential_service")),
        "sub_agents_count": len(config.get("sub_agents") or []),
        "runner_plugins_count": len(config.get("runner_plugins") or []),
        "evals": _safe_summary(config.get("evals") or []),
        "deployment_target": config.get("deployment_target"),
        "metadata": _safe_summary(config.get("metadata") or {}),
        "user_id": run_kwargs.get("user_id"),
        "session_id": run_kwargs.get("session_id"),
        "state_delta": _safe_summary(run_kwargs.get("state_delta")),
        "applied_native_agent_options": [
            option_name
            for option_name in native_agent_options
            if option_name in config and config[option_name] is not None
        ],
        "applied_native_runner_options": [
            option_name
            for option_name in native_runner_options
            if option_name in config and config[option_name] is not None
        ],
    }


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


def _response_schema(llm_request: Any) -> dict[str, Any] | None:
    config = getattr(llm_request, "config", None)
    schema = getattr(config, "response_schema", None)
    return schema if isinstance(schema, dict) else None


def _value_for_schema(schema: dict[str, Any], contents: list[Any]) -> Any:
    if "default" in schema:
        return schema["default"]
    raw_type = schema.get("type", "string")
    types = raw_type if isinstance(raw_type, list) else [raw_type]
    if "object" in types:
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        selected = list(required) or list(properties)
        return {
            name: _value_for_schema(properties.get(name, {}), contents)
            for name in selected
        }
    if "array" in types:
        return []
    if "boolean" in types:
        return True
    if "integer" in types:
        return 1
    if "number" in types:
        return 1.0
    return _last_user_text(contents) or "ok"


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


def _final_output(events: list[Any], spec: AgentSpec) -> Any:
    for event in reversed(events):
        text = _event_text(event)
        if text:
            return _coerce_structured_output(text, spec)
    return ""


def _coerce_structured_output(text: str, spec: AgentSpec) -> Any:
    if spec.output_type is None and spec.output_schema is None:
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if spec.output_type is not None:
        model_validate = getattr(spec.output_type, "model_validate", None)
        if callable(model_validate):
            return model_validate(payload)
    return payload


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
    transfer_to_agent = _event_transfer_to_agent(event)
    if transfer_to_agent:
        payload["transfer_to_agent"] = transfer_to_agent
    keys = {str(key).lower() for key in payload}
    if transfer_to_agent:
        event_type = "workflow"
    elif function_response:
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


def _run_diagnostics(
    compiled: CompiledGoogleADKAgent,
    events: list[Any],
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "events_count": len(events),
        "event_types": [type(event).__name__ for event in events],
        "text_events_count": sum(1 for event in events if _event_text(event)),
        "function_calls_count": sum(1 for event in events if _event_function_call(event)),
        "function_responses_count": sum(1 for event in events if _event_function_response(event)),
        "transfers": [
            transfer
            for event in events
            if (transfer := _event_transfer_to_agent(event))
        ],
        "session": {
            "app_name": _runner_value(compiled.runner, "app_name"),
            "user_id": run_kwargs.get("user_id"),
            "session_id": run_kwargs.get("session_id"),
            "auto_create_session": _runner_value(compiled.runner, "auto_create_session"),
            "session_service": _safe_summary(_runner_value(compiled.runner, "session_service")),
        },
        "services": {
            "memory_service": _safe_summary(_runner_value(compiled.runner, "memory_service")),
            "artifact_service": _safe_summary(_runner_value(compiled.runner, "artifact_service")),
            "credential_service": _safe_summary(_runner_value(compiled.runner, "credential_service")),
        },
        "extension": {
            "sub_agents_count": len(compiled.config.get("sub_agents") or []),
            "evals": _safe_summary(compiled.config.get("evals") or []),
            "deployment_target": compiled.config.get("deployment_target"),
        },
    }


def _runner_value(runner: Any, key: str) -> Any:
    kwargs = getattr(runner, "kwargs", None)
    if isinstance(kwargs, dict) and key in kwargs:
        return kwargs[key]
    return getattr(runner, key, None)


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


def _event_transfer_to_agent(event: Any) -> str | None:
    actions = getattr(event, "actions", None)
    if actions is None and isinstance(event, dict):
        actions = event.get("actions")
    if isinstance(actions, dict):
        transfer = actions.get("transfer_to_agent") or actions.get("transferToAgent")
    else:
        transfer = getattr(actions, "transfer_to_agent", None)
        if transfer is None:
            transfer = getattr(actions, "transferToAgent", None)
    return str(transfer) if transfer else None


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


def _safe_summary(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_safe_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_summary(item) for key, item in value.items()}
    return type(value).__name__
