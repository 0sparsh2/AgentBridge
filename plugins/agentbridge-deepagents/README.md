# agentbridge-deepagents

External AgentBridge adapter for LangChain Deep Agents.

## Adopted Version

- Native package: `deepagents`
- Adopted range: `>=0.7,<1`
- Verified API line: `0.7.19`

## Coverage

The adapter maps `AgentSpec` and `ToolSpec` to `create_deep_agent`, including
model routing, system prompts, tools, memory, skills, permissions, filesystem
backends, subagents, middleware, human approval, structured output, state and
context schemas, checkpointing, stores, caching, debug mode, and native option
pass-through. Native backend, sandbox, filesystem, and subagent objects are
preserved as escape hatches and summarized in normalized diagnostics.

Install it separately because Deep Agents may bring filesystem, sandbox, and
provider-specific dependencies:

```bash
pip install agentbridge-deepagents
```
