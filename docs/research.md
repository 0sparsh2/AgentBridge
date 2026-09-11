# AgentBridge Research Notes

## Positioning Map

| Layer | Project | What It Standardizes | AgentBridge Relationship |
| --- | --- | --- | --- |
| Model provider | LiteLLM | OpenAI-compatible access to many model providers | AgentBridge should delegate model routing instead of rebuilding this layer. |
| Tool/data access | MCP | How agents connect to tools and data sources | AgentBridge can wrap MCP-backed tools later, but v0 uses Python callables. |
| Frontend interaction | AG-UI | Agent-to-app event protocol and shared UI state | AgentBridge can emit AG-UI-shaped events from backend-normalized events. |
| Agent runtime | LangGraph | Stateful graphs, durable execution, explicit orchestration | AgentBridge should target it as a production backend. |
| Agent runtime | CrewAI | Role/task/crew-based agent prototyping | AgentBridge should target it as the prototype backend in the migration story. |
| Agent runtime | Pydantic AI | Typed Python agents, tools, and structured output | AgentBridge should target it as the typed-output backend. |

## Market Insight

The strongest wedge is not “another simpler agent framework.” The stronger wedge is:

> Define your agent once, then run, compare, and migrate it across the framework that fits the next stage of maturity.

This mirrors the pattern that made model gateways valuable: fragmentation becomes painful once teams operate more than one provider, framework, or product surface.

## MVP Implications

- Start as a Python SDK because the target frameworks are Python-native.
- Include a deterministic mock backend so the interface is always testable.
- Keep framework adapters optional to avoid forcing heavy dependencies.
- Normalize only the minimum shared concepts in v0: instructions, model, tools, input, output, events, metadata, raw backend escape hatch.
- Treat AG-UI as an output shape, not as a full server implementation.
