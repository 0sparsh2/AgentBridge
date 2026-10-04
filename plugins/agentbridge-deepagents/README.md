# agentbridge-deepagents

External AgentBridge adapter for LangChain Deep Agents.

## Adopted Version

- Native package: `deepagents`
- Adopted range: `>=0.7,<1`
- API baseline: `0.7.19`
- Install: `pip install -e 'plugins/agentbridge-deepagents[runtime]'`

## Native Surface

The adapter maps `AgentSpec` and `ToolSpec` to `create_deep_agent`, including model routing,
system prompts, tools, memory, skills, permissions, filesystem backends, subagents, middleware,
human approval, structured output, state/context schemas, checkpointing, stores, caching, debug
mode, and native option pass-through. Native backend, sandbox, filesystem, and subagent objects
are preserved as escape hatches and summarized in normalized diagnostics. Protocol labels such as
`ag_ui` and `a2a` plus sandbox metadata can be recorded without forcing a protocol or sandbox
dependency into AgentBridge core.

Heavy filesystem and sandbox dependencies remain external. The core AgentBridge package does not
install or import `deepagents`.
