"""Offline report showing the same AgentSpec's framework and AgentCore paths."""

from __future__ import annotations

from typing import Any


def build_framework_to_agentcore_report(
    *,
    agent_name: str,
    framework_results: dict[str, Any],
    agentcore_config: dict[str, Any],
) -> dict[str, Any]:
    """Compare local framework outputs with the configured AgentCore target."""

    return {
        "schema_version": "agentbridge.framework-to-agentcore.v1",
        "agent": agent_name,
        "frameworks": {
            name: {"output_present": value is not None, "output_type": type(value).__name__}
            for name, value in framework_results.items()
        },
        "agentcore": {
            "runtime_arn_present": bool(agentcore_config.get("runtime_arn")),
            "memory_id_present": bool(agentcore_config.get("memory_id")),
            "gateway_url_present": bool(agentcore_config.get("gateway_url")),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_framework_to_agentcore_report(
        agent_name="refund_agent",
        framework_results={"langchain": "local", "langgraph": "local"},
        agentcore_config={"runtime_arn": "configured"},
    ), indent=2))
