"""Run a real local LangGraph approval flow through the remote protocol bridge."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import TypedDict

from agentbridge_langchain import LangSmithAPIClient, RemoteGraphClient


class RefundState(TypedDict):
    order_id: str
    refunded: bool


def run_demo():
    """Verify local checkpoint/resume semantics using an injected SSE transport."""

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    def review(state):
        approved = interrupt({"order_id": state["order_id"], "amount": 25})
        return {"refunded": bool(approved)}

    builder = StateGraph(RefundState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    requests = []

    def serialize(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Unsupported graph payload: {type(value).__name__}")

    def transport(method, url, headers, body):
        request = json.loads(body)
        requests.append(request)
        graph_input = request["input"]
        if request.get("command") is not None:
            graph_input = Command(**request["command"])
        frames = [b'event: metadata\ndata: {"run_id":"demo-run"}\n\n']
        for mode, chunk in graph.stream(
            graph_input, config={"configurable": {"thread_id": "approval-demo"}},
            stream_mode=["updates", "values"],
        ):
            data = json.dumps(chunk, default=serialize)
            frames.append(f"event: {mode}\ndata: {data}\n\n".encode())
        frames.append(b'event: end\ndata: null\n\n')
        return 200, {"content-type": "text/event-stream"}, b"".join(frames)

    remote = RemoteGraphClient(LangSmithAPIClient(api_key="offline", transport=transport))
    initial = remote.run(
        thread_id="approval-demo", assistant_id="refund-agent", input={"order_id": "A123"}
    )
    interrupt_id = initial.metadata["interrupts"][0]["id"]
    resumed = remote.run(
        thread_id="approval-demo", assistant_id="refund-agent", input=None,
        command={"resume": {interrupt_id: True}},
    )
    return {
        "execution": "native local LangGraph with injected remote SSE transport",
        "initial": initial.model_dump(mode="json"),
        "resumed": resumed.model_dump(mode="json"),
        "requests": requests,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
