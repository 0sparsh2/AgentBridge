"""Stream deployed graphs, or run a deterministic no-key wire-format demo."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, closing

from agentbridge_langchain import LangSmithAPIClient, RemoteGraphClient


def demo_transport(method, url, headers, body):
    """Use the same SSE framing as Agent Server without network calls."""

    return 200, {"content-type": "text/event-stream"}, (
        b'event: metadata\nid: demo-0\ndata: {"run_id":"demo"}\n\n'
        b'event: messages\nid: demo-1\ndata: {"content":"Checking refund policy"}\n\n'
        b'event: values\nid: demo-2\ndata: {"eligible":true,"order_id":"A123"}\n\n'
        b'event: end\nid: demo-3\ndata: null\n\n'
    )


async def print_async(remote, run, max_events):
    async with aclosing(remote.astream(**run)) as events:
        index = 0
        async for event in events:
            print(event.model_dump_json(), flush=True)
            index += 1
            if max_events is not None and index >= max_events:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Agent Server URL; omitting this uses the offline demo")
    parser.add_argument("--assistant", default="agent")
    parser.add_argument("--thread", default="demo-thread")
    parser.add_argument("--input", default="Check order A123 for refund eligibility.")
    parser.add_argument("--async", dest="use_async", action="store_true")
    parser.add_argument("--max-events", type=int, help="Close the stream after this many events")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")
    api = (
        LangSmithAPIClient(base_url=args.url)
        if args.url else LangSmithAPIClient(api_key="offline", transport=demo_transport)
    )
    remote = RemoteGraphClient(api)
    run = {
        "thread_id": args.thread,
        "assistant_id": args.assistant,
        "input": {"messages": [{"role": "user", "content": args.input}]},
    }
    if args.use_async:
        asyncio.run(print_async(remote, run, args.max_events))
    else:
        with closing(remote.stream(**run)) as events:
            for index, event in enumerate(events, start=1):
                print(event.model_dump_json(), flush=True)
                if args.max_events is not None and index >= args.max_events:
                    break


if __name__ == "__main__":
    main()
