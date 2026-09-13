# Capability Coverage Strategy

AgentBridge eventually needs to cover the nuance of every major agent framework without pretending all frameworks are the same. The right approach is capability-aware compatibility, not one giant universal abstraction.

## Principle

Every feature should land in one of four buckets:

- `core`: supported through the shared AgentBridge API.
- `capability`: supported only when a backend advertises that capability.
- `extension`: supported through a backend-specific extension namespace.
- `native_only`: intentionally exposed through raw backend objects because abstraction would be misleading.

## Capability Areas

AgentBridge should track framework support across these areas:

- Agent definition: instructions, role, goal, system prompt, persona, model selection.
- Tools: Python callables, JSON schema, async tools, MCP tools, tool approval, tool error handling.
- Structured output: Pydantic models, JSON schema, validation, retries on invalid output.
- Workflow: graphs, tasks, crews, handoffs, state machines, conditional routing.
- State and memory: session state, short-term memory, long-term memory, checkpoints.
- Streaming: token streaming, message deltas, tool-call lifecycle, custom events.
- Human-in-the-loop: approval gates, interrupts, resume, review queues.
- Runtime behavior: retries, timeouts, cancellation, parallelism, durable execution.
- Observability: traces, spans, logs, run diagnostics, eval hooks, cost/usage reporting.
- Deployment: local execution, hosted runtimes, serverless, containers, background jobs.
- UI protocols: AG-UI event conversion, generative UI, frontend tool calls.
- Multi-agent behavior: roles, teams, swarms, delegation, agent-to-agent protocols.
- Modalities: text, image, audio, files, browser/computer use.

## Adapter Coverage Contract

Each adapter should eventually expose:

```python
capabilities = {
    "tools.sync": "full",
    "tools.async": "partial",
    "structured_output.pydantic": "full",
    "workflow.graph": "native_only",
    "human_approval": "unsupported",
}
```

Coverage values:

- `full`: supported through AgentBridge without backend-specific code.
- `partial`: supported with documented limits.
- `extension`: supported through a backend-specific AgentBridge extension.
- `native_only`: available through the backend raw object only.
- `unsupported`: not supported by the adapter or backend.

## Capability Matrix Command

AgentBridge exposes the current adapter coverage as a machine-readable or human-readable matrix:

```bash
agentbridge capability-matrix --json
agentbridge capability-matrix --markdown
agentbridge capability-matrix --backend mock --backend langgraph --json
```

Use this command whenever adapter capabilities change. It is the first version of the future coverage report described in the roadmap.

The canonical taxonomy currently lives in `agentbridge.capabilities.CANONICAL_CAPABILITIES`. If an adapter reports a feature that is not part of the canonical taxonomy, run:

```bash
agentbridge capability-matrix --include-unknown --markdown
```

Then decide whether the feature belongs in the common taxonomy, a backend extension namespace, or native-only documentation.

## Coverage Report Command

Use the detailed coverage report when reviewing a framework adapter, plugin PR, or migration target:

```bash
agentbridge coverage-report --backend mock --backend langgraph
agentbridge coverage-report --backend langgraph --json
agentbridge coverage-report --backend langgraph --markdown
```

The report includes:

- Adapter source, such as built-in, entry point, or environment plugin.
- Adopted package name, adopted version range, installed version, and support status.
- Every advertised feature and its support level.
- Feature notes from the adapter.
- Counts by support status.
- Extension namespace metadata when the backend has AgentBridge-specific config.
- Native-only features that should be documented as escape hatches instead of flattened into the common API.

This command is stricter than the matrix. The matrix answers “how do these backends compare across the canonical taxonomy?” The coverage report answers “what exactly does this backend expose today, from which package version, and where are the framework-specific edges?”

## Conformance Checks

Adapter authors should run:

```bash
agentbridge conformance --backend my_backend
```

The conformance runner verifies the minimum AgentBridge contract:

- Capability metadata is present.
- A basic agent can compile and run.
- Streaming returns normalized events.
- Sync tools work when `tools.sync` is marked `full`.
- Structured output works when `structured_output` is marked `full`.

Passing conformance does not mean the adapter is production complete. It means the adapter satisfies the baseline compatibility contract and can be compared consistently with other backends.

## Extension Example: LangGraph

LangGraph-specific graph configuration should live under the LangGraph extension namespace instead of bloating `AgentSpec`:

```python
from agentbridge.extensions.langgraph import LangGraphExtension

agent = LangGraphExtension.with_config(
    agent,
    node_name="refund_node",
    graph_name="refund_graph",
    include_context_in_output=True,
    enable_checkpointing=True,
    route_on_context_key="intent",
    routes={"refund": "refund_node", "billing": "billing_node"},
)
```

The adapter advertises checkpointing and routing as `extension`, because they are available through LangGraph-specific configuration rather than the portable core API.

## Extension Example: Pydantic AI

Pydantic AI-specific validation and runtime settings should live under the Pydantic AI extension namespace:

```python
from agentbridge.extensions.pydantic_ai import PydanticAIExtension

agent = PydanticAIExtension.with_config(
    agent,
    retries=2,
    tool_timeout=5,
    metadata={"owner": "support"},
)
```

The adapter advertises validation retries as `extension`, because retry behavior is meaningful for Pydantic AI but not identical across all frameworks.

## Extension Example: CrewAI

CrewAI-specific role/task/crew behavior should live under the CrewAI extension namespace:

```python
from agentbridge.extensions.crewai import CrewAIExtension

agent = CrewAIExtension.with_config(
    agent,
    role="Refund specialist",
    goal="Resolve refund requests using support policy.",
    task_description="Review the customer request: {input}",
    expected_output="A refund decision with rationale.",
    process="hierarchical",
    allow_delegation=True,
    memory=True,
    human_input=True,
)
```

The external CrewAI plugin advertises delegation, long-term memory, and human approval as extension-level behavior because those concepts are native to CrewAI and should not be forced into the portable core model.

## Structured Output Contract

SDK users can provide `AgentSpec.output_type` for runtime-native typed output. If the type exposes `model_json_schema()`, AgentBridge derives `AgentSpec.output_schema` automatically.

Static manifests cannot carry Python classes, so they use `output_schema` directly:

```yaml
name: typed_agent
instructions: Return structured output.
model: openai/gpt-5
output_schema:
  type: object
  properties:
    answer:
      type: string
```

When a manifest declares `output_schema`, `agentbridge compare` and `agentbridge validate` automatically add `structured_output` to required capabilities.

## Production Tool Registry

Static manifests reference tools by name. They should not import arbitrary Python by themselves.

Production apps should expose an explicit registry:

```python
from agentbridge import ToolRegistry


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"found:{order_id}"


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(lookup_order)
    return registry
```

Then pass it to the CLI:

```bash
agentbridge run \
  --manifest examples/refund_agent.yaml \
  --backend mock \
  --tool-registry my_app.tools:build_registry \
  --input "Customer was double charged"
```

This keeps manifests declarative while giving production systems a clear allowlist for executable tools.

## Design Implication

AgentBridge should grow as a layered bridge:

```text
AgentBridge Common API
  -> Capability-aware APIs
  -> Backend-specific extensions
  -> Native raw backend escape hatch
```

This avoids two traps:

- Too shallow: a lowest-common-denominator wrapper that serious teams outgrow.
- Too abstract: a fake universal model that hides framework strengths and creates confusing behavior.

## Near-Term Implementation Path

1. Keep `BackendCapabilities` and `CapabilityStatus` updated as adapters deepen.
2. Add contract tests for every backend capability marked `full`.
3. Add backend-specific extension namespaces for features that cannot fit the common core.
4. Run `agentbridge conformance --all` before merging capability claims.
5. Expand docs with a matrix for CrewAI, LangGraph, Pydantic AI, Google ADK, Strands, OpenAI Agents SDK, AgentCore, smolagents, AutoGen/AG2, and LlamaIndex Workflows.

## Current External Adapter Wave

The current external adapter wave expands framework coverage through plugins, keeping the core install path light:

- `agentbridge-openai-agents`: OpenAI Agents SDK handoffs, guardrails, approval interruption diagnostics, app-owned approval request stores, tracing, and runner/result normalization.
- `agentbridge-strands`: Strands Agent/tool mapping, MCP clients, hook/intervention/guardrail lifecycle events, structured output, run diagnostics, tracing, and structured AWS/AgentCore deployment metadata.
- `agentbridge-langchain`: direct LangChain agent compatibility for tools, middleware, callbacks, native memory/retriever examples, stream events, run diagnostics, gated LangSmith smoke coverage, and tracing, complementing the built-in LangGraph adapter.
- `agentbridge-google-adk`: ADK sessions, memory/artifact services, sub-agent transfer diagnostics, eval labels/runner bindings, run diagnostics, and structured deployment metadata.

Each plugin should start scaffolded, advertise only honest capability statuses, then graduate individual capabilities to `full` only when contract tests exercise the native framework version documented in `docs/version_policy.md`.
