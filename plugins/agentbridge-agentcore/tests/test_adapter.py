from __future__ import annotations

from io import BytesIO

from agentbridge import AgentSpec, RunInput
from agentbridge_agentcore import AgentCoreExtension
from agentbridge_agentcore.adapter import Adapter
from agentbridge_agentcore.client import AgentCoreClient


class FakeClient:
    def __init__(self):
        self.calls = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": BytesIO(b'{"output": "approved"}')}


class FakeSession:
    def __init__(self, client):
        self.native = client

    def client(self, service, region_name=None):
        assert service == "bedrock-agentcore"
        assert region_name == "us-west-2"
        return self.native


def test_agentcore_runtime_invocation_normalizes_result_and_bindings():
    native = FakeClient()
    client = AgentCoreClient(region="us-west-2", session=FakeSession(native))
    spec = AgentCoreExtension.with_config(
        AgentSpec(name="refund", instructions="Resolve refunds.", model="bedrock/native"),
        runtime_arn="arn:runtime",
        region="us-west-2",
        memory_id="memory-123",
        gateway_url="https://gateway.example",
        identity_provider="cognito",
        code_interpreter={"enabled": True},
        browser={"enabled": False},
        observability={"trace": True},
    )
    compiled = Adapter().compile(spec)
    compiled = compiled.__class__(compiled.spec, compiled.config, client)
    result = Adapter().run(compiled, RunInput(input="A123", session_id="s-1"))

    assert result.output == "approved"
    assert result.metadata["memory_id"] == "memory-123"
    assert native.calls[0]["runtimeSessionId"] == "s-1"
    payload = native.calls[0]["payload"].decode()
    assert '"identity_provider": "cognito"' in payload
    assert '"enabled": true' in payload


def test_agentcore_observability_environment_is_secret_free():
    env = AgentCoreExtension.observability_environment(
        service_name="refund",
        log_group="/aws/bedrock-agentcore/runtimes/refund",
        runtime_id="runtime/refund",
    )
    assert env["AGENT_OBSERVABILITY_ENABLED"] == "true"
    assert env["OTEL_RESOURCE_ATTRIBUTES"].startswith("service.name=refund")
    assert "AWS_SECRET_ACCESS_KEY" not in env
