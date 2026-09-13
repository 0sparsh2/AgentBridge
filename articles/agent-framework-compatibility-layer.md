# The Agent Framework Compatibility Layer

AgentBridge is built around one idea:

> App code should talk to an agent compatibility layer, not directly to one agent framework forever.

This article explains the technical shape of that layer.

## The Small Portable Core

The portable API starts with a few framework-neutral types:

- `AgentSpec`: the agent definition.
- `ToolSpec`: Python callable tools with schema metadata.
- `RunInput`: input text, context, metadata, and session ID.
- `RunResult`: normalized output, backend name, events, usage, metadata, and raw backend result.
- `AgentEvent`: normalized message, tool, workflow, error, and completion events.
- `BackendAdapter`: the compile/run/stream/resume interface.

The flow is intentionally simple:

```text
User Code
  -> AgentSpec
  -> Adapter Registry
  -> BackendAdapter.compile()
  -> BackendAdapter.run() or stream()
  -> RunResult / AgentEvent
```

This is enough to let an app define a useful agent once and then run it against different backends.

## Why Not Put Everything In AgentSpec?

Because agent frameworks are not identical.

If `AgentSpec` tried to contain every feature from every framework, it would become confusing and unstable. It would also encourage fake portability: fields that appear universal but only make sense in one backend.

Instead, AgentBridge uses three layers:

- Portable core fields for common behavior.
- Capability metadata for explicit support claims.
- Framework extension namespaces for native nuance.

For example, a portable `AgentSpec` can define instructions, tools, model, metadata, and output schema. LangGraph-specific routing belongs in `LangGraphExtension`. LangChain callbacks and retriever hints belong in `LangChainExtension`. Strands deployment metadata belongs in `StrandsExtension`. Google ADK session and eval labels belong in `GoogleADKExtension`.

This keeps the core small while still letting the project cover the nuance of each framework over time.

## Capabilities Are The Contract

Every adapter reports capabilities such as:

- `tools.sync`
- `structured_output`
- `workflow.graph`
- `workflow.routing`
- `workflow.handoffs`
- `state.checkpointing`
- `observability.diagnostics`
- `deployment.serverless`

Each capability has a support level:

- `full`: supported through AgentBridge and backed by tests.
- `partial`: supported with known limits.
- `extension`: available through a backend-specific extension namespace.
- `native_only`: available through raw backend objects, not abstracted.
- `unsupported`: not supported.

This makes backend differences inspectable instead of hidden.

For example:

```bash
agentbridge capability-matrix --backend langgraph --backend langchain --markdown
agentbridge coverage-report --backend openai_agents --markdown
agentbridge extensions strands --json
```

## Conformance Keeps Claims Honest

A compatibility layer can only be trusted if claims are tested.

AgentBridge includes a lightweight conformance runner:

```bash
agentbridge conformance --all
```

The runner checks baseline execution, streaming, sync tools when advertised as `full`, structured output when advertised as `full`, and run diagnostics when advertised as `full`.

This gives adapter authors a minimum bar before they claim compatibility.

## Plugins Keep The Core Lightweight

The core package should not force every user to install every agent framework.

Heavy, fast-moving, blocked, or provider-specific integrations should live outside the core install path as adapter plugins. That is why integrations such as OpenAI Agents, Strands, direct LangChain, Google ADK, and CrewAI live under `plugins/`.

This gives the project two useful properties:

- The core SDK remains small and stable.
- Framework-specific dependencies can move at their own pace.

## Native Escape Hatches Matter

AgentBridge does not try to hide raw backend objects.

`RunResult.raw` preserves native results. Adapter metadata preserves runtime summaries. Extension configs can pass native objects where appropriate.

This is important because advanced users will eventually need backend-specific behavior:

- LangGraph checkpoint internals.
- LangChain callback managers.
- OpenAI Agents run state.
- Google ADK session services.
- Strands MCP clients.

The bridge should standardize the common path while preserving access to the native path.

## Where This Goes

The long-term direction is a framework compatibility ecosystem:

- More adapter plugins.
- More capability-specific contract tests.
- Migration helpers for existing apps.
- AG-UI server examples.
- Observability and eval integrations.
- Deployment metadata and eventually deployment tooling where it makes sense.

The important part is the discipline: support must be explicit, tested, and documented.

See the [Architecture](../docs/architecture.md), [Capability Coverage Strategy](../docs/capability_coverage.md), and [Plugin Authoring Guide](../docs/plugin_authoring.md) for the technical details.
