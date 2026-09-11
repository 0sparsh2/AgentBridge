"""Manifest validation helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentbridge.compare import compare_backends
from agentbridge.manifest import AgentManifest
from agentbridge.tool_registry import ToolRegistry


class ManifestValidation(BaseModel):
    """Validation result for a manifest."""

    manifest: str
    valid: bool
    required_features: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)
    backend_fit: dict[str, str] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)


def validate_manifest(
    manifest: AgentManifest,
    *,
    tool_registry: ToolRegistry | None = None,
    backends: list[str] | None = None,
) -> ManifestValidation:
    """Validate manifest tool references and backend capability fit."""

    required_features = list(manifest.required_capabilities)
    if manifest.tools and "tools.sync" not in required_features:
        required_features.append("tools.sync")
    if manifest.output_schema and "structured_output" not in required_features:
        required_features.append("structured_output")
    if "agent.instructions" not in required_features:
        required_features.append("agent.instructions")

    registered_tools = set(tool_registry.list()) if tool_registry else set()
    missing_tools = [tool.name for tool in manifest.tools if tool.name not in registered_tools]

    comparisons = compare_backends(required_features, backends=backends)
    backend_fit = {comparison.backend: comparison.fit for comparison in comparisons}
    messages: list[str] = []

    if missing_tools:
        messages.append(f"Missing tools: {', '.join(missing_tools)}")
    for comparison in comparisons:
        if comparison.missing:
            messages.append(
                f"{comparison.backend} is missing capabilities: {', '.join(comparison.missing)}"
            )

    return ManifestValidation(
        manifest=manifest.name,
        valid=not missing_tools,
        required_features=required_features,
        missing_tools=missing_tools,
        backend_fit=backend_fit,
        messages=messages,
    )
