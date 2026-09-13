# Adapter Target Research

This document tracks next adapter targets beyond the first v0 backends. It should be updated whenever new framework research changes adapter priority, package names, or capability expectations.

Research date: 2026-09-11.
Package index check: 2026-09-11.

## Summary

Recommended priority:

1. `agentbridge-openai-agents`
2. `agentbridge-strands`
3. `agentbridge-langchain`
4. `agentbridge-google-adk`

Rationale:

- OpenAI Agents SDK maps closely to AgentBridge's current concepts: agents, tools, handoffs, guardrails, runner/results, tracing, and human approval.
- Strands is important for AWS-native production agents, hooks, MCP, conversation managers, structured output, observability, and AgentCore/Lambda-style deployment paths.
- Direct LangChain support complements the built-in LangGraph adapter for teams with existing LangChain agents, middleware, callbacks, memory, retrievers, and LangSmith-style observability.
- Google ADK is strategically important for enterprise-scale, multi-language, deployment-oriented agent systems and has strong session/memory/deployment concepts.

## Target Matrix

| Target | Package Target | Distribution | Priority | Why |
| --- | --- | --- | --- | --- |
| OpenAI Agents SDK | `agentbridge-openai-agents` | External plugin | P0 | Close conceptual fit with AgentBridge events, tools, handoffs, guardrails, tracing, approvals. |
| Strands Agents | `agentbridge-strands` | External plugin | P1 | AWS-native production agent path with hooks, MCP, structured output, observability, and AgentCore alignment. |
| LangChain | `agentbridge-langchain` | External plugin | P1 | Direct compatibility for existing LangChain agent apps beyond the built-in LangGraph adapter. |
| Google ADK | `agentbridge-google-adk` | External plugin | P2 | Strong enterprise, deployment, session/memory, multi-agent story across Google ecosystem. |

## Adopted Version Targets

| Framework | Native Package | Adopted Range | Latest Observed | Status |
| --- | --- | --- | --- | --- |
| OpenAI Agents SDK | `openai-agents` | `>=0.20,<0.21` | `0.22.2`; compatible baseline `0.20.0` | Partial native adapter |
| Strands Agents | `strands-agents` | `>=1.55,<2` | `1.55.1` | Partial native adapter |
| LangChain | `langchain` | `>=1.4,<2` | `1.4.0` | Partial native adapter |
| Google ADK | `google-adk` | `>=2.9,<3` | `2.9.0` | Partial native adapter |

## OpenAI Agents SDK

Sources:

- [OpenAI Agents SDK docs](https://openai.github.io/)
- [OpenAI Agents on developers.openai.com](https://developers.openai.com/)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)

Observed concepts:

- Agent definitions.
- Runner-driven execution.
- Tools.
- Handoffs.
- Guardrails.
- Results and state.
- Integrations and observability.
- Human approvals and durable/session integrations.

AgentBridge mapping:

| AgentBridge Area | OpenAI Agents Mapping |
| --- | --- |
| `AgentSpec.instructions` | Agent instructions. |
| `ToolSpec` | SDK tools/function tools. |
| `AgentEvent` | Runner events, tool calls, handoffs, guardrails, result lifecycle. |
| `structured_output` | Agent/output type support where available. |
| `human_approval` | Approval/human-in-the-loop flows. |
| `observability.raw` | Native traces and run objects. |
| `workflow.delegation` | Handoffs. |

First plugin shape:

```bash
agentbridge scaffold-plugin plugins/agentbridge-openai-agents --backend openai_agents
```

Initial extension namespace target:

```python
OpenAIAgentsExtension.config(
    handoffs=[...],
    guardrails=[...],
    tracing=True,
    approval_policy={...},
)
```

Acceptance path:

- Compile simple `AgentSpec` into SDK Agent.
- Map `ToolSpec` callables to SDK tools.
- Run with the SDK runner and normalize final result.
- Normalize tool-call and completion events.
- Preserve raw run/result objects.
- Add conformance tests with mocked/offline model paths if available.

Risks:

- The SDK may evolve quickly.
- Some runtime features may require OpenAI credentials or hosted services.
- Handoffs and guardrails should be extension-level before becoming common API.
- `openai-agents` 0.22.x requires `openai>=3`; current AgentBridge core uses LiteLLM with `openai<3`, so executable adapter work starts on `0.20.x`.

## Google ADK

Sources:

- [ADK official site](https://adk.dev/)
- [ADK get started](https://adk.dev/get-started/)
- [google/adk-python](https://github.com/google/adk-python)
- [Google Cloud ADK overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)
- [Google Cloud ADK state and memory article](https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk)

Observed concepts:

- Code-first agent framework.
- Agents and tools.
- Debug, evaluate, and deploy flows.
- Multi-agent systems.
- Sessions and memory.
- Enterprise deployment and Google ecosystem alignment.
- Multiple language SDKs.

AgentBridge mapping:

| AgentBridge Area | Google ADK Mapping |
| --- | --- |
| `AgentSpec.instructions` | Agent instruction/prompt definition. |
| `ToolSpec` | ADK tools. |
| `state.session` | Sessions. |
| `memory.long_term` | ADK memory/storage options. |
| `workflow.delegation` | Multi-agent orchestration. |
| `observability.raw` | Native ADK run/session objects. |
| `deployment` | ADK deployment surfaces. |

First plugin shape:

```bash
agentbridge scaffold-plugin plugins/agentbridge-google-adk --backend google_adk
```

Initial extension namespace target:

```python
GoogleADKExtension.config(
    app_name="support",
    session_service="memory",
    memory_service="memory",
    artifact_service="local",
    sub_agents=[...],
)
```

Acceptance path:

- Compile simple `AgentSpec` into an ADK agent.
- Map sync `ToolSpec` callables into ADK tool functions.
- Support session id propagation from `RunInput.session_id`.
- Preserve native session/run output as raw.
- Add capability metadata for sessions and memory.

Risks:

- ADK spans multiple languages and deployment targets; Python plugin should stay focused first.
- Session/memory services may require environment-specific setup.
- Google ecosystem deployment features should be extension-level.
- ADK `2.9.0` currently pins older OpenTelemetry packages than Strands, so combined native adapter installs may require isolated environments.

## LangChain

Sources:

- [LangChain Python docs](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain agents docs](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain middleware docs](https://docs.langchain.com/oss/python/langchain/middleware)

Observed concepts:

- Agents built on LangGraph.
- Tool calling and tool strategies.
- Middleware.
- Model/provider abstraction.
- Short-term memory and persistence patterns.
- Retrieval and RAG chains.
- Callbacks, tracing, and LangSmith ecosystem.

AgentBridge mapping:

| AgentBridge Area | LangChain Mapping |
| --- | --- |
| `AgentSpec.instructions` | System prompt / agent prompt template. |
| `ToolSpec` | LangChain tools. |
| `tools.async` | Async tool and runnable support. |
| `state.memory` | LangChain memory/checkpointer patterns. |
| `observability.tracing` | Callbacks and LangSmith-style tracing. |
| `workflow.graph` | Prefer built-in LangGraph backend for graph-native orchestration. |

First plugin shape:

```bash
agentbridge scaffold-plugin plugins/agentbridge-langchain --backend langchain
```

Initial extension namespace target:

```python
LangChainExtension.config(
    agent_type="tool_calling",
    middleware=[...],
    callbacks=[...],
    memory="conversation_buffer",
    retrievers=[...],
)
```

Acceptance path:

- Compile simple `AgentSpec` into a direct LangChain agent.
- Map `ToolSpec` callables into LangChain tools.
- Support middleware and callbacks through extension config.
- Preserve raw agent/result/callback metadata.
- Clearly distinguish direct LangChain support from the built-in LangGraph adapter.

Risks:

- LangChain agents are themselves built on LangGraph, so boundaries between `langchain` and `langgraph` adapters must stay clear.
- Model provider behavior may overlap with LiteLLM; AgentBridge should not build a second model abstraction.
- Retrieval/memory integrations should stay extension-level until common semantics are proven.

## Strands Agents

Sources:

- [Strands Agents official site](https://strandsagents.com/)
- [Strands structured output docs](https://strandsagents.com/)
- [AWS Strands Agents technical deep dive](https://aws.amazon.com/)

Observed concepts:

- Python and TypeScript SDKs.
- `Agent` and `tool` decorators.
- Conversation managers.
- MCP clients/tools.
- Hooks for tool lifecycle and guardrails.
- Structured output.
- Observability and trace attributes.
- AWS/Bedrock/AgentCore deployment alignment.

AgentBridge mapping:

| AgentBridge Area | Strands Mapping |
| --- | --- |
| `AgentSpec.instructions` | `system_prompt`. |
| `ToolSpec` | `@tool` or tool objects. |
| `structured_output` | `structured_output_model` / structured output schema. |
| `streaming.events` | Hook/event lifecycle. |
| `human_approval` | Hook-based interrupt/approval patterns. |
| `state.session` | Conversation managers. |
| `observability.raw` | Trace attributes and native agent/run objects. |
| `tools.mcp` | MCP client tooling. |

First plugin shape:

```bash
agentbridge scaffold-plugin plugins/agentbridge-strands --backend strands
```

Initial extension namespace target:

```python
StrandsExtension.config(
    conversation_manager="sliding_window",
    trace_attributes={"service": "support-agent"},
    hooks=[...],
    mcp_clients=[...],
)
```

Acceptance path:

- Compile `AgentSpec` into Strands `Agent`.
- Map `ToolSpec` callables into Strands tools.
- Map `output_type` to structured output path.
- Normalize hook/tool lifecycle events into `AgentEvent`.
- Preserve trace attributes and raw result.

Risks:

- Python and TypeScript APIs may diverge.
- AWS deployment integrations should remain extension-level.
- Hook-driven control flow may need a richer AgentBridge event model.

## Capability Gaps To Add Before These Adapters

- `tools.mcp`: added to canonical taxonomy.
- `tools.openapi`: added to canonical taxonomy.
- `guardrails`: added to canonical taxonomy.
- `workflow.handoffs`: added to canonical taxonomy.
- `deployment.serverless`: added to canonical taxonomy.
- `observability.tracing`: added to canonical taxonomy.
- `observability.diagnostics`: added to canonical taxonomy.
- `state.memory`: added to canonical taxonomy.
- `runtime.retries`: added to canonical taxonomy.
- `evals`: added to canonical taxonomy.

## Recommended Next Work

1. Add OpenAI Agents backend-neutral approval resume queues and deeper real-SDK approval fixtures around SDK run state.
2. Add Strands end-to-end MCP server/client fixtures and deeper guardrail behavior tests beyond normalized hook/intervention/guardrail lifecycle event shapes.
3. Add LangChain LangSmith smoke tests when configured and more real-world Runnable/compiled-graph import fixtures beyond the native retriever/checkpointer/store example.
4. Add Google ADK native session/memory/artifact service behavior fixtures, eval execution hooks, and deployment publishing examples beyond metadata preservation.
5. Keep import/migration helpers for existing LangChain and LangGraph apps conservative, expanding only when object shapes can be inspected safely.
6. Use `agentbridge conformance --all`, coverage reports, extension schemas, and issue checklists as the release gate before claiming broader framework parity.

Current migration helper status:

- `import_langchain_agent()` extracts obvious name, model, prompt, tools, runnable methods, LCEL steps, graph summaries, schema names, interrupt hints, native storage/checkpoint hints, and extension hints from LangChain-like objects.
- `import_langgraph_graph()` extracts graph names and visible node names while marking topology as native-only/manual-review territory.
