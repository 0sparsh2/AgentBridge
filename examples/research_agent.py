"""Demonstrate tool use and normalized streaming events."""

from __future__ import annotations

from agentbridge import AgentSpec, ToolSpec, stream_agent


def web_search(query: str) -> str:
    """Search the web for a query."""

    return f"Mock search result for: {query}"


agent = AgentSpec(
    name="research_agent",
    instructions="Research a topic and cite the most useful findings.",
    model="openai/gpt-5",
    tools=[ToolSpec.from_function(web_search)],
)

for event in stream_agent(agent, framework="mock", input="agent framework interoperability"):
    print(event.model_dump())
