"""Static agent manifest loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from agentbridge.tool_registry import ToolRegistry
from agentbridge.types import AgentSpec


class ManifestTool(BaseModel):
    """A tool reference inside a static manifest."""

    name: str = Field(min_length=1)


class AgentManifest(BaseModel):
    """Serializable agent definition for CLI and future migration workflows."""

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend_config: dict[str, Any] = Field(default_factory=dict)
    tools: list[ManifestTool] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)

    @field_validator("name", "instructions", "model")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    def to_agent_spec(self, tool_registry: ToolRegistry | None = None) -> AgentSpec:
        """Convert a manifest into an AgentSpec."""

        resolved_tools = []
        if self.tools:
            if tool_registry is None:
                raise ValueError(
                    "Manifest tools require a ToolRegistry. "
                    "Pass tool_registry=... or use the CLI built-in demo registry."
                )
            resolved_tools = [tool_registry.get(tool.name) for tool in self.tools]

        return AgentSpec(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=resolved_tools,
            metadata=self.metadata,
            backend_config=self.backend_config,
        )


def load_manifest(path: str | Path) -> AgentManifest:
    """Load an agent manifest from JSON, YAML, or YML."""

    manifest_path = Path(path)
    raw = manifest_path.read_text()
    suffix = manifest_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    else:
        raise ValueError("Unsupported manifest extension. Use .json, .yaml, or .yml.")

    if not isinstance(data, dict):
        raise ValueError("Agent manifest must be an object.")
    return AgentManifest.model_validate(data)


def load_agent_spec(path: str | Path, tool_registry: ToolRegistry | None = None) -> AgentSpec:
    """Load an AgentSpec from a static manifest."""

    return load_manifest(path).to_agent_spec(tool_registry=tool_registry)
