"""AgentBridge adapter for an already deployed AgentCore Runtime."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from agentbridge.adapters import BackendAdapter
from agentbridge.types import AgentEvent, AgentSpec, BackendCapabilities, RunInput, RunResult

from .client import AgentCoreClient
from .config import AgentCoreConfig


@dataclass(frozen=True)
class CompiledAgentCore:
    spec: AgentSpec
    config: AgentCoreConfig
    client: AgentCoreClient


class Adapter(BackendAdapter):
    backend_name = "agentcore"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            features={
                "agent.runtime": "full",
                "deployment.runtime": "full",
                "state.session": "full",
                "memory.agentcore": "extension",
                "tools.mcp": "extension",
                "observability.tracing": "extension",
                "protocols.a2a": "extension",
                "protocols.ag_ui": "extension",
            },
            notes={
                "agent.runtime": "Invokes an existing Bedrock AgentCore Runtime through the AWS SDK.",
                "deployment.runtime": "Deployment publishing remains an AgentCore CLI/IaC concern; runtime invocation is normalized.",
                "memory.agentcore": "Preserves AgentCore Memory IDs in the binding and payload contract.",
                "tools.mcp": "Gateway URL and native tool bindings remain available to the deployed runtime.",
                "observability.tracing": "Trace IDs and observability bindings are forwarded without storing secrets.",
            },
        )

    def compile(self, spec: AgentSpec) -> CompiledAgentCore:
        config = AgentCoreConfig.model_validate(spec.backend_config.get(self.backend_name, {}))
        return CompiledAgentCore(spec=spec, config=config, client=AgentCoreClient(region=config.region))

    def run(self, compiled: CompiledAgentCore, run_input: RunInput) -> RunResult:
        session_id = run_input.session_id or f"agentbridge-{compiled.spec.name}"
        payload = {
            "input": run_input.input,
            "context": run_input.context,
            "agent": compiled.spec.name,
            "memory_id": compiled.config.memory_id,
            "gateway_url": compiled.config.gateway_url,
            "identity_provider": compiled.config.identity_provider,
            "code_interpreter": compiled.config.code_interpreter,
            "browser": compiled.config.browser,
        }
        raw = compiled.client.invoke_runtime(
            runtime_arn=compiled.config.runtime_arn,
            session_id=session_id,
            payload=payload,
            qualifier=compiled.config.qualifier,
            trace_id=run_input.metadata.get("trace_id"),
        )
        output = raw.get("output", raw) if isinstance(raw, dict) else raw
        events = [
            AgentEvent(type="message", backend=self.backend_name, data={"content": output}),
            AgentEvent(type="complete", backend=self.backend_name, data={"output": output}),
        ]
        return RunResult(
            output=output,
            backend=self.backend_name,
            events=events,
            metadata={
                "runtime_arn": compiled.config.runtime_arn,
                "session_id": session_id,
                "memory_id": compiled.config.memory_id,
                "gateway_url": compiled.config.gateway_url,
                "identity_provider": compiled.config.identity_provider,
                "code_interpreter": compiled.config.code_interpreter,
                "browser": compiled.config.browser,
                "observability": compiled.config.observability,
            },
            raw=raw,
        )

    def stream(self, compiled: CompiledAgentCore, run_input: RunInput) -> Iterator[AgentEvent]:
        yield from self.run(compiled, run_input).events
