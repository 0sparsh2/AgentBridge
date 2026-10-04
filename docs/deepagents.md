# Deep Agents Integration

AgentBridge exposes LangChain Deep Agents through the optional external
`agentbridge-deepagents` plugin. The plugin keeps Deep Agents' filesystem, sandbox,
interpreter, and deployment dependencies outside the core install while preserving
the native option surface.

## Version Policy

| Package | Adopted range | API baseline | Boundary |
| --- | --- | --- | --- |
| `deepagents` | `>=0.7,<1` | `0.7.19` | External runtime plugin |

Install it only when needed:

```bash
pip install -e "plugins/agentbridge-deepagents[runtime]"
```

## Portable Definition

```python
from agentbridge import AgentSpec, ToolSpec, run_agent

agent = AgentSpec(
    name="research_agent",
    instructions="Research the request and cite the files you inspect.",
    model="openai/gpt-5",
    tools=[ToolSpec.from_function(search_catalog)],
    backend_config={
        "deepagents": {
            "skills": ["/workspace/skills/research/"],
            "memory": ["/workspace/memory/AGENTS.md"],
            "permissions": [{"mode": "deny", "path": "/workspace/secrets/**"}],
            "backend": filesystem_backend,
            "native_options": {"custom_profile": "research"},
        }
    },
)

result = run_agent(agent, backend="deepagents", input="Find the refund policy.")
```

## Native Surface Mapping

The adapter forwards these `create_deep_agent` options from
`agent.backend_config["deepagents"]`: `middleware`, `subagents`, `skills`, `memory`,
`permissions`, `backend`, `checkpointer`, `store`, `interrupt_on`, `response_format`,
`context_schema`, and `general_purpose_subagent`. A native object is not converted or
reimplemented; it is passed through unchanged. Other compatible options can be supplied
with `native_options`, with collisions against portable fields rejected explicitly.

Run context is passed through Deep Agents' native `context=` channel. Session IDs become
runtime `thread_id` configuration, and observability metadata/callbacks use the same
AgentBridge configuration helpers as the LangChain and LangGraph adapters.

## Boundary And Tests

The plugin has credential-free contract tests for model normalization, tool translation,
native option forwarding, context propagation, and normalized streaming. Live provider
execution, sandbox security, filesystem persistence, remote deployment, and hosted tracing
still require the corresponding credentials or service fixtures. Those are intentionally
separate smoke lanes rather than pretending that a local fake proves a hosted integration.

```bash
uv run pytest -q plugins/agentbridge-deepagents/tests
uv run ruff check plugins/agentbridge-deepagents/agentbridge_deepagents plugins/agentbridge-deepagents/tests
```
