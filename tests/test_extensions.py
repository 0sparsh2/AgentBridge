from __future__ import annotations

import pytest

from agentbridge.extensions import FrameworkExtension, UnsupportedExtension
from agentbridge.extensions.crewai import CrewAIExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.pydantic_ai import PydanticAIExtension
from agentbridge.extensions.registry import extension_profile, extension_profiles


def test_framework_extension_preserves_raw_escape_hatch() -> None:
    raw = object()
    extension = FrameworkExtension(raw=raw)

    assert extension.require_raw() is raw


def test_framework_extension_requires_raw_when_missing() -> None:
    with pytest.raises(UnsupportedExtension, match="raw backend object"):
        FrameworkExtension().require_raw()


def test_framework_specific_extension_namespaces_exist() -> None:
    assert LangGraphExtension.framework == "langgraph"
    assert PydanticAIExtension.framework == "pydantic_ai"
    assert CrewAIExtension.framework == "crewai"


def test_extension_native_placeholders_fail_clearly() -> None:
    with pytest.raises(UnsupportedExtension, match="conditional routing"):
        LangGraphExtension().conditional_routing()


def test_langgraph_extension_builds_serializable_config() -> None:
    config = LangGraphExtension.config(
        node_name="refund_node",
        graph_name="refund_graph",
        include_context_in_output=True,
        enable_checkpointing=True,
    )

    assert config == {
        "node_name": "refund_node",
        "graph_name": "refund_graph",
        "include_context_in_output": True,
        "enable_checkpointing": True,
    }


def test_pydantic_ai_extension_builds_serializable_config() -> None:
    config = PydanticAIExtension.config(
        retries=2,
        tool_timeout=5,
        metadata={"owner": "support"},
        custom_output_args={"eligible": True, "reason": "ok"},
    )

    assert config == {
        "retries": 2,
        "tool_timeout": 5.0,
        "metadata": {"owner": "support"},
        "custom_output_args": {"eligible": True, "reason": "ok"},
    }


def test_crewai_extension_builds_serializable_config() -> None:
    config = CrewAIExtension.config(
        role="Refund specialist",
        goal="Resolve refund requests",
        backstory="Expert in support policy.",
        task_description="Review {input}",
        expected_output="Refund decision",
        process="hierarchical",
        verbose=True,
        allow_delegation=True,
        memory=True,
        human_input=True,
        metadata={"owner": "support"},
    )

    assert config == {
        "role": "Refund specialist",
        "goal": "Resolve refund requests",
        "backstory": "Expert in support policy.",
        "task_description": "Review {input}",
        "expected_output": "Refund decision",
        "process": "hierarchical",
        "verbose": True,
        "allow_delegation": True,
        "memory": True,
        "human_input": True,
        "metadata": {"owner": "support"},
    }


def test_extension_registry_lists_framework_profiles() -> None:
    profiles = {profile.framework: profile for profile in extension_profiles()}

    assert {"langgraph", "pydantic_ai", "crewai"}.issubset(profiles)
    assert "node_name" in profiles["langgraph"].config_schema["properties"]
    assert "retries" in profiles["pydantic_ai"].config_schema["properties"]
    assert "role" in profiles["crewai"].config_schema["properties"]


def test_extension_profile_returns_one_framework() -> None:
    profile = extension_profile("crewai")

    assert profile.framework == "crewai"
    assert "workflow.roles_tasks" in profile.capabilities
