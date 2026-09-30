"""Show the same AgentSpec shape targeting an existing AgentCore Runtime."""

from agentbridge import AgentSpec, run_agent
from agentbridge_agentcore import AgentCoreExtension


agent = AgentCoreExtension.with_config(
    AgentSpec(
        name="refund_agent",
        instructions="Decide whether a customer is eligible for a refund.",
        model="bedrock/native",
    ),
    runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/refund",
    region="us-west-2",
    memory_id="memory-123",
    gateway_url="https://gateway.example",
    observability={"trace": True},
)


if __name__ == "__main__":
    result = run_agent(agent, backend="agentcore", input="Check order A123")
    print(result.output)
