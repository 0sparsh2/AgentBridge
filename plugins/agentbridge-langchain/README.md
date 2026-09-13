# agentbridge-langchain

AgentBridge adapter plugin for `langchain`.

## Adopted Framework Version

- Native package: `langchain`
- Adopted range: `>=1.4,<2`
- Verified locally: `1.4.0`
- Status: partial native adapter

## Target Capabilities

- Direct LangChain `create_agent` compatibility alongside the built-in LangGraph backend.
- `ToolSpec` to `StructuredTool` mapping.
- Structured output through native LangChain `response_format` and typed `structured_response`.
- Middleware, callbacks, memory hints, retriever hints, and native `create_agent` options through
  `LangChainExtension`.
- LangSmith/callback-style tracing metadata via native LangChain runtime config.
- Streaming normalization for LangChain `stream_events(..., version="v3")` event envelopes and
  `stream(..., stream_mode=["messages", "updates", "custom"], version="v2")` chunks.

## Install

```bash
pip install -e .
```

## Verify Discovery

```bash
agentbridge plugins
agentbridge list-backends
agentbridge inspect-backend langchain --json
agentbridge conformance --backend langchain
```

The conformance runner uses the plugin-only `agentbridge/offline` model string. That path still
builds a LangChain `create_agent` graph and executes it natively, but uses a tiny local
`BaseChatModel` implementation so contract checks do not require provider packages or API keys.

## LangChain vs LangGraph

Use `langchain` when you are migrating or standardizing an existing LangChain agent app that depends
on `create_agent`, `StructuredTool`, middleware, callbacks, memory, retrievers, or LangSmith-style
instrumentation.

Use `langgraph` when the app is primarily a durable workflow: explicit graph topology, checkpoints,
interrupts, resumability, state routing, or production orchestration are the core requirement.
LangChain agents are built on LangGraph internally, but AgentBridge keeps these adapters separate so
teams can choose between direct app compatibility and explicit graph control.

When `LangChainExtension.config(callbacks=..., metadata=...)` is used, the adapter passes those
values into LangChain's native runtime config and includes a serializable `runtime_config` summary in
`RunResult.metadata`.

When `stream_agent(..., backend="langchain")` is used, the adapter first tries LangChain's event
streaming API and normalizes message deltas, tool-call chunks, completed tool calls, tool results,
and custom updates into `AgentEvent` values. If a LangChain runtime only supports classic
`stream()`, the adapter requests `messages`, `updates`, and `custom` stream modes before falling back
to the runtime's default stream signature.

When `LangChainExtension.config(memory=..., retrievers=...)` is used, AgentBridge records those hints
in `RunResult.metadata["extension_summary"]`. Portable memory/retriever semantics are not claimed
unless the caller supplies native LangChain objects such as `checkpointer` and `store`, which the
adapter forwards directly into `create_agent`.

## Local Development Without Installing

```bash
export AGENTBRIDGE_ADAPTER_PLUGINS="agentbridge_langchain.adapter"
agentbridge plugins
```

## Implementation Checklist

- Add the framework package dependency to `pyproject.toml`.
- Update `Adapter.capabilities()` with honest support metadata.
- Implement `compile()` by translating `AgentSpec` to native framework objects.
- Implement `run()` by returning a normalized `RunResult`.
- Implement `stream()` if the backend supports streaming.
- Add contract tests for every capability marked `full`.
- Run `agentbridge conformance --backend langchain` before publishing.
- Document adopted and verified framework versions.
