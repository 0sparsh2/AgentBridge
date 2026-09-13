"""Show how framework-specific nuance stays outside the portable AgentSpec core."""

from __future__ import annotations

import json

from agentbridge import AgentSpec
from agentbridge.extensions.google_adk import GoogleADKExtension
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge.extensions.openai_agents import OpenAIAgentsExtension
from agentbridge.extensions.strands import StrandsExtension


base_agent = AgentSpec(
    name="refund_agent",
    instructions="Handle refund requests with policy checks and safe escalation.",
    model="openai/gpt-5",
)


variants = {
    "langchain": LangChainExtension.with_config(
        base_agent,
        callbacks=["langsmith"],
        metadata={"team": "support"},
        memory="conversation_buffer",
        retrievers=["refund_policy_docs"],
        interrupt_before=["tools"],
    ),
    "openai_agents": OpenAIAgentsExtension.with_config(
        base_agent,
        handoff_description="Escalate billing disputes.",
        approval_policy={"issue_refund": "required"},
        conversation_id="support-thread-1",
        tracing=True,
    ),
    "strands": StrandsExtension.with_config(
        base_agent,
        mcp_clients=["orders_mcp"],
        guardrails=["refund_policy"],
        deployment_target="agentcore",
        deployment={
            "runtime": "bedrock-agentcore",
            "entrypoint": "app:agent",
            "region": "us-east-1",
        },
    ),
    "google_adk": GoogleADKExtension.with_config(
        base_agent,
        app_name="support_app",
        sub_agents=["billing_agent"],
        evals=["refund_quality_eval"],
        deployment_target="vertex_ai",
        deployment={
            "runtime": "adk",
            "entrypoint": "app:agent",
            "region": "us-central1",
        },
    ),
}


for framework, agent in variants.items():
    print(f"\n{framework}")
    print(json.dumps(agent.backend_config[framework], indent=2, sort_keys=True))
