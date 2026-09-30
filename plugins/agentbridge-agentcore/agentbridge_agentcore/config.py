"""Framework-neutral configuration for AgentCore service bindings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.types import AgentSpec


class AgentCoreConfig(BaseModel):
    """AgentCore Runtime target plus optional service bindings.

    Service bindings are descriptive/native escape hatches. They are preserved
    so the same AgentSpec can be deployed with Runtime, Memory, Gateway,
    Identity, and observability configuration without making AWS a core dep.
    """

    runtime_arn: str = Field(min_length=1)
    region: str | None = None
    qualifier: str | None = None
    memory_id: str | None = None
    gateway_url: str | None = None
    identity_provider: str | None = None
    code_interpreter: dict[str, Any] = Field(default_factory=dict)
    browser: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)
    native_options: dict[str, Any] = Field(default_factory=dict)


class AgentCoreExtension:
    """Attach AgentCore bindings to an AgentSpec without importing boto3."""

    @staticmethod
    def config(**kwargs: Any) -> dict[str, Any]:
        return AgentCoreConfig.model_validate(kwargs).model_dump(exclude_none=True)

    @staticmethod
    def with_config(spec: AgentSpec, **kwargs: Any) -> AgentSpec:
        backend_config = dict(spec.backend_config)
        backend_config["agentcore"] = AgentCoreExtension.config(**kwargs)
        return spec.model_copy(update={"backend_config": backend_config})

    @staticmethod
    def observability_environment(
        *,
        service_name: str,
        log_group: str,
        runtime_id: str,
        trace_exporter: str = "otlp",
    ) -> dict[str, str]:
        """Return safe ADOT environment settings for AgentCore telemetry."""

        return {
            "AGENT_OBSERVABILITY_ENABLED": "true",
            "OTEL_PYTHON_DISTRO": "aws_distro",
            "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
            "OTEL_RESOURCE_ATTRIBUTES": (
                f"service.name={service_name},"
                f"aws.log.group.names={log_group},cloud.resource_id={runtime_id}"
            ),
            "OTEL_TRACES_EXPORTER": trace_exporter,
        }
