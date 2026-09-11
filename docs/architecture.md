# AgentBridge Architecture

## Core Flow

```text
User Code
  -> AgentSpec
  -> Adapter Registry
  -> BackendAdapter.compile()
  -> BackendAdapter.run() or stream()
  -> RunResult / AgentEvent
```

## Core Types

- `AgentSpec`: framework-neutral agent definition.
- `ToolSpec`: Python callable wrapper with JSON-schema-like input metadata.
- `RunInput`: input text plus context, metadata, and session id.
- `AgentEvent`: normalized event for messages, tool calls, tool results, errors, and completion.
- `RunResult`: normalized result with output, backend, events, usage, metadata, and raw backend result.
- `BackendCapabilities`: feature coverage metadata advertised by each backend.
- `BackendAdapter`: framework adapter contract.
- `AgentManifest`: static JSON/YAML agent definition used by the CLI and future migration tooling.

## Compatibility Philosophy

AgentBridge should not flatten every framework into the lowest common denominator. The long-term architecture has three layers:

- Common core: features that every backend can reasonably support, such as instructions, model, tools, input, output, and normalized events.
- Capability layer: feature flags that describe what a backend supports, such as graph state, typed output, memory, human approval, retries, tracing, multimodal input, or multi-agent handoffs.
- Native escape hatches: backend-specific configuration and raw objects for features that cannot be expressed cleanly in the common model yet.

This lets AgentBridge expose a simple API without hiding the unique strengths of LangGraph, CrewAI, Pydantic AI, Google ADK, Strands, OpenAI Agents SDK, AgentCore, smolagents, and future frameworks.

## Adapter Responsibilities

- Translate `AgentSpec` into backend-native objects.
- Translate `ToolSpec` into backend-native tools.
- Normalize final outputs into `RunResult`.
- Normalize streaming behavior into `AgentEvent`.
- Keep raw backend objects available for users who need escape hatches.
- Publish backend capability metadata so users can inspect fit before running or migrating.

## Registry

AgentBridge exposes a small adapter registry so backends can be looked up by name:

- Built in: `mock`, `pydantic_ai`, `langgraph`.
- Custom adapters can be registered with `register_adapter()`.
- Optional dependencies are imported lazily by each adapter.
- External adapters can be loaded from the `agentbridge.adapters` entry point group or from `AGENTBRIDGE_ADAPTER_PLUGINS`.

## Manifests and CLI

AgentBridge supports static JSON/YAML manifests for no-code agent definitions that can be run from the CLI. v0 manifests intentionally do not execute tool declarations because arbitrary Python callables cannot be safely deserialized from static files. Tool manifests will require a registry before they become executable.

Manifest tools now resolve only through an explicit `ToolRegistry`, which keeps static files safe while allowing named, pre-registered tools. The CLI ships with deterministic demo tools for examples; production apps should provide their own registry.

The `validate` command checks that manifest tools exist in a registry and compares required capabilities against selected backends before a run or migration attempt.

## Adapter Plugins

Heavy or blocked frameworks should move into external adapter packages instead of forcing all users to install their dependency trees. Plugin packages register a `BackendAdapter` subclass through the `agentbridge.adapters` entry point group. Local development plugins can be loaded by setting `AGENTBRIDGE_ADAPTER_PLUGINS` to comma-separated module names.

See `docs/plugin_authoring.md` for the plugin package contract.

## AG-UI Event Mapping

AgentBridge does not implement an AG-UI server in v0. It provides a converter from `AgentEvent` to AG-UI-shaped dictionaries so users can pipe normalized backend events into a frontend protocol later.
