import asyncio

from agentbridge import AgentSpec, astream_agent


def test_async_streaming_returns_normalized_events():
    async def collect():
        return [
            event
            async for event in astream_agent(
                AgentSpec(name="async", instructions="Reply.", model="agentbridge/offline"),
                backend="mock",
                input="hello",
            )
        ]

    events = asyncio.run(collect())
    assert events[-1].type == "complete"
    assert events[-1].backend == "mock"
