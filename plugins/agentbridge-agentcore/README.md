# AgentBridge AgentCore Plugin

Optional bridge for Amazon Bedrock AgentCore Runtime and service bindings.

## Adopted surface

- Runtime invocation through `bedrock-agentcore.invoke_agent_runtime`.
- Native operation escape hatch for Runtime, Memory, Gateway, Identity, and future AgentCore APIs.
- AgentCore Memory IDs, Gateway URLs, A2A/AG-UI labels, trace IDs, and observability settings preserved in normalized AgentBridge metadata.
- AgentCore Observability trace and session propagation for CloudWatch-compatible runtime telemetry.
- Compatible with LangChain, LangGraph, Strands, CrewAI, Google ADK, and other runtimes once deployed to AgentCore.

Install with AWS support:

```bash
pip install -e 'plugins/agentbridge-agentcore[aws]'
```

```python
from agentbridge import AgentSpec, run_agent
from agentbridge_agentcore import AgentCoreExtension

agent = AgentCoreExtension.with_config(
    AgentSpec(name="refund", instructions="Resolve refunds.", model="bedrock/native"),
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/refund",
    region="us-west-2",
    memory_id="memory-123",
    gateway_url="https://gateway.example",
    observability={"trace": True},
)
result = run_agent(agent, backend="agentcore", input="Check order A123")
```

AgentCore deployment, IAM, container packaging, and CloudWatch configuration remain AWS
control-plane responsibilities. AgentBridge covers the application-side invocation and keeps
native service operations available through `AgentCoreClient.call(...)`.
