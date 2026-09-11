"""Convert AgentBridge events into AG-UI-shaped dictionaries."""

from __future__ import annotations

from agentbridge import AgentSpec, event_to_agui, stream_agent


agent = AgentSpec(
    name="ui_demo_agent",
    instructions="Show normalized event conversion.",
    model="openai/gpt-5",
)

for event in stream_agent(agent, backend="mock", input="hello frontend"):
    print(event_to_agui(event))
