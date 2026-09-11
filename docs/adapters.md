# Adapter Guide

Adapters translate `AgentSpec` into backend-native runtime objects and normalize backend responses into `RunResult` and `AgentEvent`.

## Current Support Matrix

| Backend | Distribution | Install | Status | Best Fit |
| --- | --- | --- | --- | --- |
| `mock` | Core | Included | Verified | Tests, docs, CI, deterministic examples. |
| `langgraph` | Core optional extra | `pip install -e ".[langgraph]"` | Verified locally | Durable stateful orchestration and graph-shaped flows. |
| `pydantic_ai` | Core optional extra | `pip install -e ".[pydantic-ai]"` | Verified locally | Typed Python-native agents and structured output pathfinding. |
| `crewai` | External plugin scaffold | `plugins/agentbridge-crewai` | Blocked | High-level role/task/crew prototyping once dependency resolution is isolated. |

## Built-In Adapter Rules

Core adapters should be included only when they satisfy all of these:

- The dependency tree is reasonably light.
- The adapter can be tested in CI without paid API calls.
- The package resolves cleanly across supported Python versions.
- The adapter adds strategic coverage to the migration story.

If an adapter fails these rules, keep it as a plugin.

## `mock`

The mock adapter is intentionally boring. It exists so documentation, examples, CLI flows, and contract tests can run without network access or model credentials.

Use it for:

- Unit tests.
- README examples.
- Manifest validation.
- Comparing result shapes.

Do not use it to infer real model behavior.

## `langgraph`

LangGraph is the production-oriented backend target for stateful orchestration.

Current focus:

- Compile a minimal graph from `AgentSpec`.
- Execute deterministic tool-backed flows in local tests.
- Normalize graph outputs into `RunResult`.
- Preserve native raw objects for deeper graph behavior.

Next areas:

- Checkpointing and resume.
- Conditional routing.
- Richer graph state.
- Tool-call lifecycle streaming.
- Human-in-the-loop interrupts.

## `pydantic_ai`

Pydantic AI is the typed-agent target.

Current focus:

- Compile `AgentSpec` into a Pydantic AI agent path.
- Keep dependency footprint lower with `pydantic-ai-slim`.
- Test offline using Pydantic AI test utilities where possible.
- Map `AgentSpec.output_type` to Pydantic AI's native `output_type`.

Next areas:

- Pydantic model output schemas.
- Validation retry behavior.
- Typed tool argument mapping.

## `crewai`

CrewAI is intentionally outside the core install path for now.

Current status:

- Plugin scaffold exists at [plugins/agentbridge-crewai](../plugins/agentbridge-crewai).
- The adapter package should own its own dependency constraints.
- The core package reports CrewAI as an external plugin target, not a built-in backend.

Before CrewAI can be called supported:

- Resolve dependencies in a compatible Python environment.
- Add compile tests for agent, task, and crew mapping.
- Add a contract test using the same refund agent as other backends.
- Document which CrewAI features are core, extension, or native-only.

## Capability Declaration

Adapters should describe support honestly. Avoid claiming a feature is available through AgentBridge just because the native framework supports it.

Recommended status values:

- `full`: Supported through AgentBridge and covered by tests.
- `partial`: Supported with documented limits.
- `extension`: Supported through an adapter-specific extension namespace.
- `native_only`: Available only through raw backend objects.
- `unsupported`: Not supported by the adapter or backend.

See [capability_coverage.md](capability_coverage.md) for the long-term coverage model.

## Adding A New Adapter

1. Start as an external plugin unless the dependency tree is clearly core-friendly.
2. Generate the starter package with `agentbridge scaffold-plugin ./plugins/agentbridge-my-framework --backend my_framework`.
3. Implement `BackendAdapter.compile()`, `run()`, and `stream()`.
4. Add capability metadata before exposing the adapter publicly.
5. Add a contract test using a shared example agent.
6. Update [version_policy.md](version_policy.md) with adopted and verified versions.
7. Update this guide with strengths, limits, and native escape hatches.
