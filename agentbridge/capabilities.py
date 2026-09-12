"""Capability taxonomy and matrix helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentbridge.registry import inspect_backend, list_adapters


class CapabilityFeature(BaseModel):
    """A canonical capability tracked across adapters."""

    key: str
    category: str
    label: str
    description: str


class CapabilityMatrixRow(BaseModel):
    """Support status for one capability across selected backends."""

    feature: CapabilityFeature
    support: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)


class CapabilityMatrix(BaseModel):
    """A matrix of backend support for canonical capabilities."""

    backends: list[str]
    rows: list[CapabilityMatrixRow]

    def as_markdown(self) -> str:
        """Render the matrix as a Markdown table."""

        headers = ["Feature", "Category", *self.backends]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in self.rows:
            values = [
                f"`{row.feature.key}`",
                row.feature.category,
                *[row.support.get(backend, "unsupported") for backend in self.backends],
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


CANONICAL_CAPABILITIES: tuple[CapabilityFeature, ...] = (
    CapabilityFeature(
        key="agent.instructions",
        category="Agent definition",
        label="Instructions",
        description="Backend can consume agent instructions or system guidance.",
    ),
    CapabilityFeature(
        key="agent.model",
        category="Agent definition",
        label="Model routing",
        description="Backend can consume the AgentSpec model string.",
    ),
    CapabilityFeature(
        key="tools.sync",
        category="Tools",
        label="Synchronous tools",
        description="Backend can execute synchronous ToolSpec callables.",
    ),
    CapabilityFeature(
        key="tools.async",
        category="Tools",
        label="Asynchronous tools",
        description="Backend can support async tool execution semantics.",
    ),
    CapabilityFeature(
        key="tools.mcp",
        category="Tools",
        label="MCP tools",
        description="Backend can consume or expose Model Context Protocol tools.",
    ),
    CapabilityFeature(
        key="tools.openapi",
        category="Tools",
        label="OpenAPI tools",
        description="Backend can consume OpenAPI-described tools or action schemas.",
    ),
    CapabilityFeature(
        key="structured_output",
        category="Structured output",
        label="Structured output",
        description="Backend can return validated typed or schema-constrained output.",
    ),
    CapabilityFeature(
        key="structured_output.validation_retries",
        category="Structured output",
        label="Validation retries",
        description="Backend can retry when structured output validation fails.",
    ),
    CapabilityFeature(
        key="workflow.graph",
        category="Workflow",
        label="Graph workflow",
        description="Backend can represent graph-shaped orchestration.",
    ),
    CapabilityFeature(
        key="workflow.routing",
        category="Workflow",
        label="Conditional routing",
        description="Backend can route execution based on state or input.",
    ),
    CapabilityFeature(
        key="workflow.roles_tasks",
        category="Workflow",
        label="Roles and tasks",
        description="Backend can represent role/task/crew-style orchestration.",
    ),
    CapabilityFeature(
        key="workflow.delegation",
        category="Workflow",
        label="Delegation",
        description="Backend can delegate work between roles, agents, or crews.",
    ),
    CapabilityFeature(
        key="workflow.handoffs",
        category="Workflow",
        label="Handoffs",
        description="Backend can transfer execution between specialized agents.",
    ),
    CapabilityFeature(
        key="state.session",
        category="State",
        label="Session state",
        description="Backend can preserve or consume session state.",
    ),
    CapabilityFeature(
        key="state.checkpointing",
        category="State",
        label="Checkpointing",
        description="Backend can checkpoint and resume execution state.",
    ),
    CapabilityFeature(
        key="memory.long_term",
        category="State",
        label="Long-term memory",
        description="Backend can use durable or long-term memory features.",
    ),
    CapabilityFeature(
        key="state.memory",
        category="State",
        label="Memory services",
        description="Backend can attach native short-term or long-term memory services.",
    ),
    CapabilityFeature(
        key="streaming.events",
        category="Streaming",
        label="Normalized events",
        description="Backend can stream or synthesize normalized AgentEvent values.",
    ),
    CapabilityFeature(
        key="guardrails",
        category="Runtime behavior",
        label="Guardrails",
        description="Backend can enforce input, output, or tool-use guardrails.",
    ),
    CapabilityFeature(
        key="runtime.retries",
        category="Runtime behavior",
        label="Retries",
        description="Backend can configure retries around model, tool, or validation failures.",
    ),
    CapabilityFeature(
        key="human_approval",
        category="Human-in-the-loop",
        label="Human approval",
        description="Backend can support approval gates, interrupts, or review steps.",
    ),
    CapabilityFeature(
        key="observability.raw",
        category="Observability",
        label="Raw observability",
        description="Adapter preserves raw backend objects for inspection and tracing.",
    ),
    CapabilityFeature(
        key="observability.tracing",
        category="Observability",
        label="Tracing",
        description="Backend can emit traces, spans, run IDs, or trace metadata.",
    ),
    CapabilityFeature(
        key="evals",
        category="Observability",
        label="Evaluations",
        description="Backend can attach evaluation hooks, datasets, or scoring workflows.",
    ),
    CapabilityFeature(
        key="deployment.serverless",
        category="Deployment",
        label="Serverless deployment",
        description="Backend has an adapter path for serverless or managed deployment surfaces.",
    ),
    CapabilityFeature(
        key="agui.events",
        category="UI protocols",
        label="AG-UI-shaped events",
        description="Backend events can be converted into AG-UI-shaped dictionaries.",
    ),
)


def capability_matrix(
    *,
    backends: list[str] | None = None,
    include_unknown: bool = False,
) -> CapabilityMatrix:
    """Build a capability matrix for selected backends."""

    selected_backends = backends or list_adapters()
    inspected = {backend: inspect_backend(backend) for backend in selected_backends}
    feature_keys = [feature.key for feature in CANONICAL_CAPABILITIES]

    if include_unknown:
        unknown_keys = sorted(
            {
                key
                for capabilities in inspected.values()
                for key in capabilities.features
                if key not in feature_keys
            }
        )
        features = list(CANONICAL_CAPABILITIES) + [
            CapabilityFeature(
                key=key,
                category="Uncategorized",
                label=key,
                description="Feature reported by an adapter but not yet in the canonical taxonomy.",
            )
            for key in unknown_keys
        ]
    else:
        features = list(CANONICAL_CAPABILITIES)

    rows: list[CapabilityMatrixRow] = []
    for feature in features:
        support: dict[str, str] = {}
        notes: dict[str, str] = {}
        for backend, capabilities in inspected.items():
            support[backend] = capabilities.status(feature.key)
            if feature.key in capabilities.notes:
                notes[backend] = capabilities.notes[feature.key]
        rows.append(CapabilityMatrixRow(feature=feature, support=support, notes=notes))

    return CapabilityMatrix(backends=selected_backends, rows=rows)
