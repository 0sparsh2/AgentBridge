# AgentBridge Requirements

## Problem

Agent framework choice is becoming a long-term architecture decision too early in the build cycle. Teams often prototype with one framework, then discover they need a different runtime for production concerns such as state, typed outputs, tracing, governance, deployment, or UI event compatibility.

AgentBridge reduces framework lock-in by providing a framework-neutral agent specification and adapter interface.

## Target Users

- Startup teams that prototyped with a high-level framework and now need production behavior.
- Platform teams supporting multiple agent frameworks across internal product teams.
- AI engineers comparing frameworks before committing to one runtime.
- App developers who want normalized streaming and tool events regardless of backend.

## Goals

- Define an agent once with `AgentSpec`.
- Run the same spec through multiple backend adapters.
- Normalize results and stream events.
- Preserve backend escape hatches through `raw` fields.
- Make missing optional framework dependencies obvious and actionable.
- Provide enough examples to explain the migration story quickly.

## Non-Goals

- AgentBridge is not a new agent framework.
- AgentBridge does not host a gateway server in v0.
- AgentBridge does not replace AG-UI, MCP, LiteLLM, LangGraph, CrewAI, or Pydantic AI.
- AgentBridge does not automatically convert existing framework projects in v0.
- AgentBridge does not own model-provider routing; it uses LiteLLM-style model names.

## Success Criteria

- A developer can define one simple `AgentSpec`.
- The same spec can run against the included `mock` backend and compile through optional framework adapters.
- `RunResult` and `AgentEvent` have stable JSON-serializable shapes.
- Examples demonstrate backend selection, tool use, streaming, and AG-UI-shaped event conversion.
- Tests can run without paid model API calls.
- Adapter docs state adopted and verified framework versions.
