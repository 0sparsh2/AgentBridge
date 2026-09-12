# Migration Helpers

AgentBridge migration helpers inspect existing framework-native objects and produce a conservative report. The goal is to reduce migration risk, not pretend every native concept can be flattened into `AgentSpec`.

## Public Helpers

```python
from agentbridge import import_langchain_agent, import_langgraph_graph

report = import_langchain_agent(existing_agent)
if report.convertible:
    agent = report.agent_spec
```

Available helpers:

- `import_langchain_agent(obj)`: extracts obvious agent name, model, system prompt, tools, and extension hints from LangChain-like agents or compiled graph objects.
- `import_langgraph_graph(obj)`: extracts graph name and visible node names into LangGraph extension hints.

## Report Shape

Each helper returns `MigrationReport`:

- `agent_spec`: a best-effort `AgentSpec`, when safe to produce.
- `required_capabilities`: capabilities the migrated spec likely needs.
- `extension_hints`: backend-specific configuration hints.
- `native_only`: features that should remain native or require manual review.
- `findings`: human-readable notes about migration risk.

## Design Rule

When a feature is ambiguous, migration helpers should report it as `extension` or `native_only` instead of silently dropping it.

Examples:

- LangChain middleware, callbacks, memory, and retrievers become `LangChainExtension` hints.
- LangChain compiled graph topology is marked `native_only` unless it can be safely represented.
- LangGraph nodes are surfaced as extension hints, but node behavior and edge conditions still require manual review.

## Non-Goals

- No arbitrary Python imports from static manifests.
- No automatic conversion of full production graph topology yet.
- No claim that native callback, memory, retriever, checkpoint, or deployment behavior is portable without tests.
