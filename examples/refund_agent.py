"""Run the same refund agent against any AgentBridge framework adapter."""

from __future__ import annotations

import argparse

from agentbridge import AgentSpec, ToolSpec, run_agent


def check_order(order_id: str) -> str:
    """Return refund eligibility for an order."""

    return f"Order lookup for '{order_id}': eligible for refund."


def issue_refund(order_id: str) -> str:
    """Issue a refund for an eligible order."""

    return f"Refund queued for '{order_id}'."


def build_agent() -> AgentSpec:
    return AgentSpec(
        name="refund_agent",
        instructions="Decide whether a customer is eligible for a refund.",
        model="openai/gpt-5",
        tools=[
            ToolSpec.from_function(check_order),
            ToolSpec.from_function(issue_refund),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="mock", choices=["mock", "pydantic_ai", "crewai", "langgraph"])
    parser.add_argument("--input", default="Customer says order A123 was double charged.")
    args = parser.parse_args()

    result = run_agent(build_agent(), framework=args.framework, input=args.input)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
