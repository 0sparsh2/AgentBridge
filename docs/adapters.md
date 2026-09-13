# Adapter Guide

Adapters translate `AgentSpec` into backend-native runtime objects and normalize backend responses into `RunResult` and `AgentEvent`.

## Current Support Matrix

| Backend | Distribution | Install | Status | Best Fit |
| --- | --- | --- | --- | --- |
| `mock` | Core | Included | Verified | Tests, docs, CI, deterministic examples. |
| `langgraph` | Core optional extra | `pip install -e ".[langgraph]"` | Verified locally | Durable stateful orchestration and graph-shaped flows. |
| `pydantic_ai` | Core optional extra | `pip install -e ".[pydantic-ai]"` | Verified locally | Typed Python-native agents and structured output pathfinding. |
| `crewai` | External plugin scaffold | `plugins/agentbridge-crewai` | Blocked | High-level role/task/crew prototyping once dependency resolution is isolated. |

## Distribution Types

AgentBridge keeps the common SDK small and moves framework-specific dependency risk outward.

`Core` means the backend ships with the base package and has no heavy optional framework dependency.
Today this is only `mock`.

`Core optional extra` means the adapter implementation lives in the core repository, but its framework
dependency is installed only when requested. LangGraph and Pydantic AI are here because they are
strategic early backends, resolve cleanly in CI, and can be tested without paid model calls.

`External plugin` means the adapter is a separate package discovered through AgentBridge plugin
loading. This is the default for heavier or faster-moving ecosystems such as OpenAI Agents, Strands,
LangChain, and Google ADK.

`External plugin scaffold` means AgentBridge has created the package shape and extension namespace,
but the real framework dependency is blocked or not yet safely verified in this environment. CrewAI
is currently here because the targeted dependency set conflicts with this repo's current Python and
LangChain/LangSmith dependency path.

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
- Support `LangGraphExtension.config()` for node naming, graph naming, context echoing, in-memory checkpointing, and context-based conditional routing.
- Validate deterministic structured output through `AgentSpec.output_type` or `output_schema`.
- Report checkpoint-backed interrupt state when `interrupt_before` or `interrupt_after` pauses execution.
- Resume checkpointed interrupts through `resume_agent(compiled, backend="langgraph", session_id=...)`
  or `get_adapter("langgraph").resume(...)`.

Next areas:

- Richer graph state.
- Tool-call lifecycle streaming.
- Portable approval and resume helpers on top of native LangGraph interrupts.

Example:

```python
from agentbridge import AgentSpec, run_agent
from agentbridge.extensions.langgraph import LangGraphExtension

agent = LangGraphExtension.with_config(
    AgentSpec(
        name="refund_agent",
        instructions="Check refund eligibility.",
        model="openai/gpt-5",
    ),
    node_name="refund_node",
    graph_name="refund_graph",
    include_context_in_output=True,
    enable_checkpointing=True,
    interrupt_before=["refund_node"],
    route_on_context_key="intent",
    routes={"refund": "refund_node", "billing": "billing_node"},
)

result = run_agent(
    agent,
    framework="langgraph",
    input="Check order A123",
    session_id="customer-123",
)
```

If an interrupt pauses execution, `RunResult.metadata["interrupted"]` is `True` and workflow events
include `phase="interrupted"` with the next node and checkpoint identifiers. Resume requires the same
compiled graph object when using the default in-memory checkpointer, because that compiled runtime owns
the checkpoint state:

```python
from agentbridge import AgentSpec, RunInput, get_adapter, resume_agent

adapter = get_adapter("langgraph")
compiled = adapter.compile(agent)

paused = adapter.run(
    compiled,
    RunInput(input="Approve refund A123", session_id="customer-123"),
)

if paused.metadata.get("interrupted"):
    resumed = resume_agent(
        compiled,
        backend="langgraph",
        input="Approved by manager.",
        session_id="customer-123",
    )
```

Human approval remains `extension` rather than `full` because AgentBridge can pause and resume
checkpointed LangGraph execution, but does not yet provide a backend-neutral review queue or approval
policy model.

## `pydantic_ai`

Pydantic AI is the typed-agent target.

Current focus:

- Compile `AgentSpec` into a Pydantic AI agent path.
- Keep dependency footprint lower with `pydantic-ai-slim`.
- Test offline using Pydantic AI test utilities where possible.
- Map `AgentSpec.output_type` to Pydantic AI's native `output_type`.
- Support `PydanticAIExtension.config()` for retries, tool timeout, metadata, and offline test-model output controls.

Next areas:

- Pydantic model output schemas.
- Typed tool argument mapping.

Example:

```python
from agentbridge import AgentSpec, run_agent
from agentbridge.extensions.pydantic_ai import PydanticAIExtension

agent = PydanticAIExtension.with_config(
    AgentSpec(
        name="refund_decision_agent",
        instructions="Return a typed refund decision.",
        model="openai/gpt-5",
        output_type=RefundDecision,
    ),
    retries=2,
    tool_timeout=5,
    metadata={"owner": "support"},
)

result = run_agent(agent, framework="pydantic_ai", input="Customer was double charged.")
```

## `crewai`

CrewAI is intentionally outside the core install path for now.

Current status:

- Plugin scaffold exists at [plugins/agentbridge-crewai](../plugins/agentbridge-crewai).
- The adapter package should own its own dependency constraints.
- The core package reports CrewAI as an external plugin target, not a built-in backend.
- `CrewAIExtension.config()` maps role, goal, backstory, task description, expected output, process, delegation, memory, and human input into the external plugin.

Before CrewAI can be called supported:

- Resolve dependencies in a compatible Python environment.
- Add compile tests for agent, task, and crew mapping.
- Add a contract test using the same refund agent as other backends.
- Document which CrewAI features are core, extension, or native-only.

Example:

```python
from agentbridge import AgentSpec, run_agent
from agentbridge.extensions.crewai import CrewAIExtension

agent = CrewAIExtension.with_config(
    AgentSpec(
        name="refund_agent",
        instructions="Resolve refund requests.",
        model="openai/gpt-5",
    ),
    role="Refund specialist",
    goal="Resolve refund requests using support policy.",
    task_description="Review the customer request: {input}",
    expected_output="A refund decision with rationale.",
    process="hierarchical",
    allow_delegation=True,
    memory=True,
    human_input=True,
)

result = run_agent(agent, framework="crewai", input="Customer was double charged.")
```

## Capability Declaration

Adapters should describe support honestly. Avoid claiming a feature is available through AgentBridge just because the native framework supports it.

Recommended status values:

- `full`: Supported through AgentBridge and covered by tests.
- `partial`: Supported with documented limits.
- `extension`: Supported through an adapter-specific extension namespace.
- `native_only`: Available only through raw backend objects.
- `unsupported`: Not supported by the adapter or backend.

See [capability_coverage.md](capability_coverage.md) for the long-term coverage model.

## Extension Namespaces

Framework-specific nuance belongs in extension namespaces when it is useful but not portable enough for the common `AgentSpec`.

Initial namespaces:

| Namespace | Intended Nuance |
| --- | --- |
| `agentbridge.extensions.langgraph` | Checkpointing, resume, conditional routing, graph state helpers. |
| `agentbridge.extensions.pydantic_ai` | Validation retries, dependency injection, typed output helpers. |
| `agentbridge.extensions.crewai` | Crews, roles, tasks, delegation helpers. |

Extensions should preserve the native framework's mental model. They are the main path for adopting every framework's nuance without bloating the portable core.

Inspect available extension namespaces with:

```bash
agentbridge extensions
agentbridge extensions langgraph --json
```

## Adding A New Adapter

1. Start as an external plugin unless the dependency tree is clearly core-friendly.
2. Generate the starter package with `agentbridge scaffold-plugin ./plugins/agentbridge-my-framework --backend my_framework`.
3. Implement `BackendAdapter.compile()`, `run()`, and `stream()`.
4. Add capability metadata before exposing the adapter publicly.
5. Add a contract test using a shared example agent.
6. Update [version_policy.md](version_policy.md) with adopted and verified versions.
7. Update this guide with strengths, limits, and native escape hatches.
