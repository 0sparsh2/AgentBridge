# Framework Support

This page is generated from the versioned files in `frameworks/`. It records the declared support contract; runtime availability is still reported by `agentbridge coverage-report`.

## Summary

| Framework status | Count |
| --- | ---: |
| `blocked` | 1 |
| `partial` | 6 |
| `verified` | 2 |

## Integrations

| Framework | Package | Status | Verification lanes | Declared capabilities |
| --- | --- | --- | --- | ---: |
| [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html) | `agentbridge-agentcore` `>=0.1,<1` | `partial` | `offline`, `optional_dependency`, `credentialed` | 9 |
| [CrewAI](https://docs.crewai.com/) | `crewai` `>=0.11.2,<0.12` | `blocked` | `offline`, `optional_dependency`, `credentialed` | 8 |
| [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) | `deepagents` `>=0.7,<1` | `partial` | `offline`, `optional_dependency`, `credentialed` | 9 |
| [Google ADK](https://google.github.io/adk-docs/) | `google-adk` `>=2.9,<3` | `partial` | `offline`, `optional_dependency`, `credentialed` | 10 |
| [LangChain](https://docs.langchain.com/oss/python/langchain/overview) | `langchain` `>=1.4,<2` | `partial` | `offline`, `optional_dependency`, `credentialed` | 13 |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | `langgraph` `>=1.2.11,<2` | `verified` | `offline`, `optional_dependency`, `credentialed` | 13 |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | `openai-agents` `>=0.20,<0.21` | `partial` | `offline`, `optional_dependency`, `credentialed` | 10 |
| [Pydantic AI](https://ai.pydantic.dev/) | `pydantic-ai-slim` `>=2.42,<3` | `verified` | `offline`, `optional_dependency`, `credentialed` | 11 |
| [Strands Agents](https://strandsagents.com/latest/) | `strands-agents` `>=1.55,<2` | `partial` | `offline`, `optional_dependency`, `credentialed` | 11 |

## Amazon Bedrock AgentCore

- Framework ID: `agentcore`
- Plugin: `plugins/agentbridge-agentcore`
- Next slice: Add IAM-aware runtime, Memory, Gateway, and deployment smoke tests.
- Native escape hatches: `runtime client`, `memory`, `gateway`, `identity`, `native AWS operations`
- Upstream review sources: [AgentCore documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)

| Capability | Declared support |
| --- | --- |
| `agent.model` | `extension` |
| `agui.events` | `extension` |
| `deployment.serverless` | `extension` |
| `observability.diagnostics` | `extension` |
| `observability.tracing` | `extension` |
| `state.memory` | `extension` |
| `state.session` | `extension` |
| `streaming.events` | `extension` |
| `tools.mcp` | `extension` |

## CrewAI

- Framework ID: `crewai`
- Plugin: `plugins/agentbridge-crewai`
- Next slice: Verify the existing plugin in an isolated Python and dependency environment.
- Native escape hatches: `crew`, `task`, `process`, `native agent options`
- Upstream review sources: [CrewAI documentation](https://docs.crewai.com/)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `extension` |
| `agent.model` | `extension` |
| `human_approval` | `extension` |
| `observability.raw` | `extension` |
| `state.memory` | `extension` |
| `tools.sync` | `extension` |
| `workflow.delegation` | `extension` |
| `workflow.roles_tasks` | `extension` |

## LangChain Deep Agents

- Framework ID: `deepagents`
- Plugin: `plugins/agentbridge-deepagents`
- Next slice: Add credentialed sandbox and hosted deployment execution lanes.
- Native escape hatches: `sandbox`, `filesystem backend`, `subagents`, `skills`
- Upstream review sources: [Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/overview)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `extension` |
| `deployment.serverless` | `native_only` |
| `human_approval` | `extension` |
| `state.checkpointing` | `extension` |
| `state.memory` | `extension` |
| `structured_output` | `extension` |
| `tools.sync` | `extension` |
| `workflow.handoffs` | `extension` |

## Google ADK

- Framework ID: `google_adk`
- Plugin: `plugins/agentbridge-google-adk`
- Next slice: Run credentialed eval and deployment publishing conformance in an isolated telemetry environment.
- Native escape hatches: `Agent`, `Runner`, `session service`, `artifact service`, `deployment metadata`
- Upstream review sources: [Google ADK documentation](https://google.github.io/adk-docs/)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `extension` |
| `deployment.serverless` | `extension` |
| `evals` | `extension` |
| `state.memory` | `extension` |
| `state.session` | `extension` |
| `streaming.events` | `extension` |
| `structured_output` | `extension` |
| `tools.sync` | `full` |
| `workflow.handoffs` | `extension` |

## LangChain

- Framework ID: `langchain`
- Plugin: `plugins/agentbridge-langchain`
- Next slice: Graduate high-value hosted deployment, provider, and observability paths with credentialed tests.
- Native escape hatches: `native_options`, `native model`, `middleware`, `callbacks`
- Upstream review sources: [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview), [LangChain docs index](https://docs.langchain.com/llms.txt)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `extension` |
| `evals` | `extension` |
| `human_approval` | `extension` |
| `observability.tracing` | `extension` |
| `state.memory` | `extension` |
| `state.session` | `extension` |
| `streaming.events` | `extension` |
| `structured_output` | `extension` |
| `tools.async` | `extension` |
| `tools.mcp` | `extension` |
| `tools.sync` | `full` |
| `workflow.graph` | `native_only` |

## LangGraph

- Framework ID: `langgraph`
- Plugin: `agentbridge.adapters.langgraph`
- Next slice: Credentialed remote graph, deployment, and hosted runtime conformance.
- Native escape hatches: `native_graph`, `native_options`, `runtime_config`
- Upstream review sources: [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `partial` |
| `agent.model` | `extension` |
| `human_approval` | `extension` |
| `observability.diagnostics` | `full` |
| `observability.tracing` | `extension` |
| `state.checkpointing` | `extension` |
| `state.session` | `extension` |
| `streaming.events` | `full` |
| `structured_output` | `full` |
| `tools.async` | `full` |
| `tools.sync` | `full` |
| `workflow.graph` | `full` |
| `workflow.routing` | `extension` |

## OpenAI Agents SDK

- Framework ID: `openai_agents`
- Plugin: `plugins/agentbridge-openai-agents`
- Next slice: Resolve the OpenAI dependency-major conflict and verify native resume execution.
- Native escape hatches: `Agent`, `Runner`, `handoffs`, `guardrails`, `MCP servers`
- Upstream review sources: [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `extension` |
| `guardrails` | `extension` |
| `human_approval` | `extension` |
| `observability.tracing` | `extension` |
| `streaming.events` | `partial` |
| `structured_output` | `extension` |
| `tools.mcp` | `extension` |
| `tools.sync` | `full` |
| `workflow.handoffs` | `extension` |

## Pydantic AI

- Framework ID: `pydantic_ai`
- Plugin: `agentbridge.adapters.pydantic_ai`
- Next slice: Expand durable execution, provider smoke tests, and native evaluation integration.
- Native escape hatches: `agent_options`, `model_settings`, `deps`
- Upstream review sources: [Pydantic AI documentation](https://ai.pydantic.dev/)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `full` |
| `observability.diagnostics` | `full` |
| `observability.raw` | `full` |
| `runtime.retries` | `extension` |
| `state.session` | `partial` |
| `streaming.events` | `full` |
| `structured_output` | `full` |
| `structured_output.validation_retries` | `extension` |
| `tools.async` | `extension` |
| `tools.sync` | `full` |

## Strands Agents

- Framework ID: `strands`
- Plugin: `plugins/agentbridge-strands`
- Next slice: Add credentialed AWS and AgentCore deployment conformance.
- Native escape hatches: `Agent options`, `MCP client`, `hooks`, `interventions`, `AgentCore metadata`
- Upstream review sources: [Strands documentation](https://strandsagents.com/latest/)

| Capability | Declared support |
| --- | --- |
| `agent.instructions` | `full` |
| `agent.model` | `extension` |
| `deployment.serverless` | `extension` |
| `guardrails` | `extension` |
| `observability.tracing` | `extension` |
| `state.memory` | `extension` |
| `state.session` | `extension` |
| `streaming.events` | `extension` |
| `structured_output` | `extension` |
| `tools.mcp` | `extension` |
| `tools.sync` | `full` |
