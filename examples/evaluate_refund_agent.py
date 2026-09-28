"""Run a no-network AgentBridge evaluation against the mock backend."""

from agentbridge import AgentSpec, EvaluationExample, ToolSpec, evaluate_agent


def check_order(order_id: str) -> str:
    """Return a deterministic order status for the example."""

    return f"{order_id} is eligible for a refund."


agent = AgentSpec(
    name="refund_agent",
    instructions="Decide whether a customer is eligible for a refund.",
    model="agentbridge/offline",
    tools=[ToolSpec.from_function(check_order)],
)

report = evaluate_agent(
    agent,
    backend="mock",
    dataset=[
        EvaluationExample(input="Check order A123", expected_output="eligible"),
        EvaluationExample(input="Check order B456", expected_output="eligible"),
    ],
    evaluators={
        "mentions_order": lambda example, result: example.input.split()[-1] == result.output["input"].split()[-1],
    },
    dataset_name="refund-regression",
)

print(report.model_dump_json(indent=2))
