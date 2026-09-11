from __future__ import annotations

import pytest

from agentbridge.extensions import FrameworkExtension, UnsupportedExtension
from agentbridge.extensions.crewai import CrewAIExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from agentbridge.extensions.pydantic_ai import PydanticAIExtension


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
