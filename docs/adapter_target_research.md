# Adapter Target Research

This document tracks next adapter targets beyond the first v0 backends. It should be updated whenever new framework research changes adapter priority, package names, or capability expectations.

Research date: 2026-09-11.

## Summary

Recommended priority:

1. `agentbridge-openai-agents`
2. `agentbridge-google-adk`
3. `agentbridge-strands`

Rationale:

- OpenAI Agents SDK maps closely to AgentBridge's current concepts: agents, tools, handoffs, guardrails, runner/results, tracing, and human approval.
- Google ADK is strategically important for enterprise-scale, multi-language, deployment-oriented agent systems and has strong session/memory/deployment concepts.
- Strands is important for AWS-native production agents, hooks, MCP, conversation managers, structured output, observability, and AgentCore/Lambda-style deployment paths.

## Target Matrix

| Target | Package Target | Distribution | Priority | Why |
| --- | --- | --- | --- | --- |
| OpenAI Agents SDK | `agentbridge-openai-agents` | External plugin | P0 | Close conceptual fit with AgentBridge events, tools, handoffs, guardrails, tracing, approvals. |
| Google ADK | `agentbridge-google-adk` | External plugin | P1 | Strong enterprise, deployment, session/memory, multi-agent story across Google ecosystem. |
| Strands Agents | `agentbridge-strands` | External plugin | P1 | AWS-native production agent path with hooks, MCP, structured output, observability, and AgentCore alignment. |

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

- `tools.mcp`
- `guardrails`
- `workflow.handoffs`
- `deployment.serverless`
- `observability.tracing`
- `state.memory`
- `evals`

## Recommended Next Work

1. Scaffold `plugins/agentbridge-openai-agents`.
2. Add `agentbridge.extensions.openai_agents` config namespace.
3. Add capability rows for handoffs, guardrails, MCP tools, tracing, and deployment.
4. Add mocked adapter contract tests.
5. Repeat for `google_adk` and `strands` once the OpenAI Agents plugin pattern is proven.
