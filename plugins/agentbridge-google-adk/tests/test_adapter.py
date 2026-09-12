from __future__ import annotations

import sys
from types import SimpleNamespace

from agentbridge import AgentSpec, RunInput, ToolSpec
from agentbridge.extensions.google_adk import GoogleADKExtension
from agentbridge_google_adk.adapter import Adapter, CompiledGoogleADKAgent


class FakeADKAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFunctionTool:
    def __init__(self, func):
        self.func = func


class FakeSessionService:
    pass


class FakePart:
    def __init__(self, text=None):
        self.text = text

    @classmethod
    def from_text(cls, *, text):
        return cls(text=text)


class FakeContent:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class FakeRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_run_kwargs = None

    def run(self, *, user_id, session_id, new_message, state_delta=None, run_config=None):
        self.last_run_kwargs = {
            "user_id": user_id,
            "session_id": session_id,
            "new_message": new_message,
            "state_delta": state_delta,
            "run_config": run_config,
        }
        yield SimpleNamespace(
            content=FakeContent(
                role="model",
                parts=[FakePart(text=f"native google adk: {new_message.parts[0].text}")],
            )
        )


def install_fake_google_adk(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.adk.agents", SimpleNamespace(Agent=FakeADKAgent))
    monkeypatch.setitem(sys.modules, "google.adk.runners", SimpleNamespace(Runner=FakeRunner))
    monkeypatch.setitem(
        sys.modules,
        "google.adk.sessions",
        SimpleNamespace(InMemorySessionService=FakeSessionService),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.adk.tools.function_tool",
        SimpleNamespace(FunctionTool=FakeFunctionTool),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai.types",
        SimpleNamespace(Content=FakeContent, Part=FakePart),
    )


def test_adapter_compiles_and_runs_native_agent(monkeypatch) -> None:
    install_fake_google_adk(monkeypatch)
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="mock/model",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert isinstance(compiled, CompiledGoogleADKAgent)
    assert compiled.native_agent.kwargs["name"] == "support_agent"
    assert compiled.native_agent.kwargs["instruction"] == "Echo the user request."
    assert compiled.runner.kwargs["app_name"] == "support_agent"
    assert result.backend == "google_adk"
    assert result.output == "native google adk: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_forwards_google_adk_extension_surface(monkeypatch) -> None:
    install_fake_google_adk(monkeypatch)
    adapter = Adapter()
    callback = object()
    session_service = object()
    memory_service = object()
    artifact_service = object()
    credential_service = object()
    runner_plugin = object()
    sub_agent = object()
    planner = object()
    code_executor = object()
    retry_config = object()
    generate_content_config = object()
    input_schema = object()
    state_schema = object()
    agent = GoogleADKExtension.with_config(
        AgentSpec(
            name="support_agent",
            instructions="Echo the user request.",
            model="mock/model",
        ),
        app_name="support_app",
        description="Support assistant",
        global_instruction="Global policy",
        static_instruction="Static policy",
        input_schema=input_schema,
        state_schema=state_schema,
        generate_content_config=generate_content_config,
        mode="chat",
        parallel_worker=True,
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        include_contents="none",
        output_key="answer",
        planner=planner,
        code_executor=code_executor,
        retry_config=retry_config,
        timeout=12.5,
        rerun_on_resume=True,
        wait_for_output=True,
        before_agent_callback=callback,
        after_agent_callback=callback,
        before_model_callback=callback,
        after_model_callback=callback,
        on_model_error_callback=callback,
        before_tool_callback=callback,
        after_tool_callback=callback,
        on_tool_error_callback=callback,
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=artifact_service,
        credential_service=credential_service,
        runner_plugins=[runner_plugin],
        plugin_close_timeout=1.5,
        auto_create_session=False,
        sub_agents=[sub_agent],
        evals=["golden_refund_eval"],
        deployment_target="vertex_ai",
        metadata={"owner": "support"},
    )

    compiled = adapter.compile(agent)
    result = adapter.run(
        compiled,
        RunInput(
            input="hello",
            context={"tenant": "acme"},
            metadata={"user_id": "user-1"},
            session_id="session-1",
        ),
    )

    assert compiled.native_agent.kwargs["description"] == "Support assistant"
    assert compiled.native_agent.kwargs["global_instruction"] == "Global policy"
    assert compiled.native_agent.kwargs["static_instruction"] == "Static policy"
    assert compiled.native_agent.kwargs["input_schema"] is input_schema
    assert compiled.native_agent.kwargs["state_schema"] is state_schema
    assert compiled.native_agent.kwargs["generate_content_config"] is generate_content_config
    assert compiled.native_agent.kwargs["mode"] == "chat"
    assert compiled.native_agent.kwargs["parallel_worker"] is True
    assert compiled.native_agent.kwargs["disallow_transfer_to_parent"] is True
    assert compiled.native_agent.kwargs["disallow_transfer_to_peers"] is True
    assert compiled.native_agent.kwargs["include_contents"] == "none"
    assert compiled.native_agent.kwargs["output_key"] == "answer"
    assert compiled.native_agent.kwargs["planner"] is planner
    assert compiled.native_agent.kwargs["code_executor"] is code_executor
    assert compiled.native_agent.kwargs["retry_config"] is retry_config
    assert compiled.native_agent.kwargs["timeout"] == 12.5
    assert compiled.native_agent.kwargs["rerun_on_resume"] is True
    assert compiled.native_agent.kwargs["wait_for_output"] is True
    assert compiled.native_agent.kwargs["before_agent_callback"] is callback
    assert compiled.native_agent.kwargs["after_agent_callback"] is callback
    assert compiled.native_agent.kwargs["before_model_callback"] is callback
    assert compiled.native_agent.kwargs["after_model_callback"] is callback
    assert compiled.native_agent.kwargs["on_model_error_callback"] is callback
    assert compiled.native_agent.kwargs["before_tool_callback"] is callback
    assert compiled.native_agent.kwargs["after_tool_callback"] is callback
    assert compiled.native_agent.kwargs["on_tool_error_callback"] is callback
    assert compiled.native_agent.kwargs["sub_agents"] == [sub_agent]
    assert compiled.runner.kwargs["app_name"] == "support_app"
    assert compiled.runner.kwargs["session_service"] is session_service
    assert compiled.runner.kwargs["memory_service"] is memory_service
    assert compiled.runner.kwargs["artifact_service"] is artifact_service
    assert compiled.runner.kwargs["credential_service"] is credential_service
    assert compiled.runner.kwargs["plugins"] == [runner_plugin]
    assert compiled.runner.kwargs["plugin_close_timeout"] == 1.5
    assert compiled.runner.kwargs["auto_create_session"] is False
    assert compiled.runner.last_run_kwargs["user_id"] == "user-1"
    assert compiled.runner.last_run_kwargs["session_id"] == "session-1"
    assert compiled.runner.last_run_kwargs["state_delta"] == {"tenant": "acme"}
    assert result.metadata["extension_summary"] == {
        "app_name": "support_app",
        "session_service": "object",
        "memory_service": "object",
        "artifact_service": "object",
        "credential_service": "object",
        "sub_agents_count": 1,
        "runner_plugins_count": 1,
        "evals": ["golden_refund_eval"],
        "deployment_target": "vertex_ai",
        "metadata": {"owner": "support"},
        "user_id": "user-1",
        "session_id": "session-1",
        "state_delta": {"tenant": "acme"},
        "applied_native_agent_options": [
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
        ],
        "applied_native_runner_options": [
            "session_service",
            "memory_service",
            "artifact_service",
            "credential_service",
            "runner_plugins",
            "plugin_close_timeout",
            "auto_create_session",
        ],
    }


def test_adapter_runs_offline_model_through_native_runner() -> None:
    adapter = Adapter()
    spec = AgentSpec(
        name="support_agent",
        instructions="Echo the user request.",
        model="agentbridge/offline",
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="hello"))

    assert result.backend == "google_adk"
    assert result.output == "offline response: hello"
    assert [event.type for event in result.events] == ["message", "complete"]


def test_adapter_runs_offline_tool_loop_through_native_runner() -> None:
    adapter = Adapter()

    def lookup_order(order_id: str) -> str:
        """Look up an order."""

        return f"found:{order_id}"

    spec = AgentSpec(
        name="support_agent",
        instructions="Use the lookup tool.",
        model="agentbridge/offline",
        tools=[ToolSpec.from_function(lookup_order)],
    )

    compiled = adapter.compile(spec)
    result = adapter.run(compiled, RunInput(input="A123"))

    event_types = [event.type for event in result.events]
    assert result.output == "offline tool result: found:A123"
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert event_types[-1] == "complete"
