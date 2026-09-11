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
- Observability: traces, spans, logs, eval hooks, cost/usage reporting.
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
4. Expand docs with a matrix for CrewAI, LangGraph, Pydantic AI, Google ADK, Strands, OpenAI Agents SDK, AgentCore, smolagents, AutoGen/AG2, and LlamaIndex Workflows.
