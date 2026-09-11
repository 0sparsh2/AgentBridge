# AgentBridge

AgentBridge is a Python SDK for defining an agent once and running it across multiple agent frameworks.

The first MVP focuses on a migration wedge:

> Move from a fast prototype framework to a production-oriented agent runtime without rewriting tools, prompts, results, and event handling.

AG-UI standardizes how user-facing apps interact with agents. LiteLLM standardizes model-provider access. AgentBridge standardizes the app-to-agent-framework layer.

## Quickstart

```python
from agentbridge import AgentSpec, ToolSpec, run_agent


def check_order(order_id: str) -> str:
    """Return the refund status for an order."""
    return f"Order {order_id} is eligible for a refund."


agent = AgentSpec(
    name="refund_agent",
    instructions="Decide whether a customer is eligible for a refund.",
    model="openai/gpt-5",
    tools=[ToolSpec.from_function(check_order)],
)

result = run_agent(
    agent,
    backend="mock",
    input="Customer says order A123 was double charged.",
)

print(result.output)
```

The `mock` backend is included so docs, examples, and tests can run without API keys. Production backends are exposed as optional adapters:

- `pydantic_ai`
- `langgraph`

Install extras as needed:

```bash
pip install -e ".[pydantic-ai]"
pip install -e ".[langgraph]"
```

Heavy adapters live as plugins. For example, the blocked CrewAI scaffold lives in `plugins/agentbridge-crewai`.

Adopted framework versions are tracked in [docs/version_policy.md](docs/version_policy.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## CLI

```bash
agentbridge list-backends
agentbridge inspect-backend langgraph --json
agentbridge run --manifest examples/refund_agent.yaml --backend mock --input "Customer was double charged" --json
agentbridge compare --manifest examples/refund_agent.yaml --backend mock --backend langgraph --json
agentbridge validate --manifest examples/refund_agent.yaml --backend mock --backend langgraph --json
agentbridge versions --json
agentbridge plugins --json
```

## Adapter Plugins

Heavy or isolated adapters can live outside the core package. AgentBridge loads plugins from the `agentbridge.adapters` entry point group and from comma-separated module names in `AGENTBRIDGE_ADAPTER_PLUGINS`.
Plugins cannot overwrite an existing backend unless `AGENTBRIDGE_ADAPTER_PLUGIN_OVERRIDES` names the backend explicitly.

An environment plugin module can expose either `Adapter` or `get_adapter()`:

```python
from agentbridge.adapters import BackendAdapter


class Adapter(BackendAdapter):
    backend_name = "custom"
```

See [docs/plugin_authoring.md](docs/plugin_authoring.md) for the full plugin contract.

## Status

This repository is an MVP foundation. The core abstractions, docs, mock backend, AG-UI event mapping, examples, and optional adapter skeletons are present. Framework-specific adapters intentionally fail with clear dependency messages when their optional packages are not installed.
