"""Google ADK-specific extension helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.extensions.base import FrameworkExtension
from agentbridge.types import AgentSpec


class GoogleADKConfig(BaseModel):
    """AgentBridge config for Google ADK-native behavior."""

    model_config = {"arbitrary_types_allowed": True}

    app_name: str | None = None
    description: str | None = None
    global_instruction: Any | None = None
    static_instruction: Any | None = None
    input_schema: Any | None = None
    state_schema: Any | None = None
    generate_content_config: Any | None = None
    mode: str | None = None
    parallel_worker: bool | None = None
    disallow_transfer_to_parent: bool | None = None
    disallow_transfer_to_peers: bool | None = None
    include_contents: str | None = None
    output_key: str | None = None
    planner: Any | None = None
    code_executor: Any | None = None
    retry_config: Any | None = None
    timeout: float | None = None
    rerun_on_resume: bool | None = None
    wait_for_output: bool | None = None
    before_agent_callback: Any | None = None
    after_agent_callback: Any | None = None
    before_model_callback: Any | None = None
    after_model_callback: Any | None = None
    on_model_error_callback: Any | None = None
    before_tool_callback: Any | None = None
    after_tool_callback: Any | None = None
    on_tool_error_callback: Any | None = None
    session_service: Any | None = None
    memory_service: Any | None = None
    artifact_service: Any | None = None
    credential_service: Any | None = None
    runner_plugins: list[Any] = Field(default_factory=list)
    plugin_close_timeout: float | None = None
    auto_create_session: bool | None = None
    sub_agents: list[Any] = Field(default_factory=list)
    evals: list[Any] = Field(default_factory=list)
    eval_runner: Any | None = None
    capture_service_snapshots: bool | None = None
    deployment_target: str | None = None
    deployment: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoogleADKExtension(FrameworkExtension):
    """Extension namespace for Google ADK behavior."""

    framework = "google_adk"

    @staticmethod
    def config(
        *,
        app_name: str | None = None,
        description: str | None = None,
        global_instruction: Any | None = None,
        static_instruction: Any | None = None,
        input_schema: Any | None = None,
        state_schema: Any | None = None,
        generate_content_config: Any | None = None,
        mode: str | None = None,
        parallel_worker: bool | None = None,
        disallow_transfer_to_parent: bool | None = None,
        disallow_transfer_to_peers: bool | None = None,
        include_contents: str | None = None,
        output_key: str | None = None,
        planner: Any | None = None,
        code_executor: Any | None = None,
        retry_config: Any | None = None,
        timeout: float | None = None,
        rerun_on_resume: bool | None = None,
        wait_for_output: bool | None = None,
        before_agent_callback: Any | None = None,
        after_agent_callback: Any | None = None,
        before_model_callback: Any | None = None,
        after_model_callback: Any | None = None,
        on_model_error_callback: Any | None = None,
        before_tool_callback: Any | None = None,
        after_tool_callback: Any | None = None,
        on_tool_error_callback: Any | None = None,
        session_service: Any | None = None,
        memory_service: Any | None = None,
        artifact_service: Any | None = None,
        credential_service: Any | None = None,
        runner_plugins: list[Any] | None = None,
        plugin_close_timeout: float | None = None,
        auto_create_session: bool | None = None,
        sub_agents: list[Any] | None = None,
        evals: list[Any] | None = None,
        eval_runner: Any | None = None,
        capture_service_snapshots: bool | None = None,
        deployment_target: str | None = None,
        deployment: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build serializable Google ADK adapter configuration."""

        return GoogleADKConfig(
            app_name=app_name,
            description=description,
            global_instruction=global_instruction,
            static_instruction=static_instruction,
            input_schema=input_schema,
            state_schema=state_schema,
            generate_content_config=generate_content_config,
            mode=mode,
            parallel_worker=parallel_worker,
            disallow_transfer_to_parent=disallow_transfer_to_parent,
            disallow_transfer_to_peers=disallow_transfer_to_peers,
            include_contents=include_contents,
            output_key=output_key,
            planner=planner,
            code_executor=code_executor,
            retry_config=retry_config,
            timeout=timeout,
            rerun_on_resume=rerun_on_resume,
            wait_for_output=wait_for_output,
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            before_model_callback=before_model_callback,
            after_model_callback=after_model_callback,
            on_model_error_callback=on_model_error_callback,
            before_tool_callback=before_tool_callback,
            after_tool_callback=after_tool_callback,
            on_tool_error_callback=on_tool_error_callback,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            credential_service=credential_service,
            runner_plugins=runner_plugins or [],
            plugin_close_timeout=plugin_close_timeout,
            auto_create_session=auto_create_session,
            sub_agents=sub_agents or [],
            evals=evals or [],
            eval_runner=eval_runner,
            capture_service_snapshots=capture_service_snapshots,
            deployment_target=deployment_target,
            deployment=deployment or {},
            metadata=metadata or {},
        ).model_dump(exclude_none=True, exclude_defaults=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        """Return a copy of an AgentSpec with Google ADK config applied."""

        backend_config = dict(spec.backend_config)
        backend_config["google_adk"] = GoogleADKExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})
