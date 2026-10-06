"""Stream deployed graphs, or run a deterministic no-key wire-format demo."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, closing

from agentbridge_langchain import LangSmithAPIClient, RemoteGraphClient


def demo_transport(method, url, headers, body):
    """Use the same SSE framing as Agent Server without network calls."""

    frames = [
        b'event: metadata\nid: demo-0\ndata: {"run_id":"demo"}\n\n',
        b'event: messages\nid: demo-1\ndata: {"content":"Checking refund policy"}\n\n',
        b'event: values\nid: demo-2\ndata: {"eligible":true,"order_id":"A123"}\n\n',
        b'event: end\nid: demo-3\ndata: null\n\n',
    ]
    last_id = headers.get("Last-Event-ID")
    if last_id is not None:
        ids = [f"demo-{index}" for index in range(len(frames))]
        frames = frames[ids.index(last_id) + 1:]
    return 200, {"content-type": "text/event-stream"}, b"".join(frames)


async def print_async(remote, run, max_events, reconnect=False):
    source = remote.areconnect(**run) if reconnect else remote.astream(**run)
    async with aclosing(source) as events:
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
    parser.add_argument("--stream-mode", nargs="+", help="Native stream modes, such as values messages")
    parser.add_argument("--resumable", action="store_true", help="Retain run events for reconnecting")
    parser.add_argument("--checkpoint", help="Replay from this checkpoint with no new graph input")
    parser.add_argument("--run-id", help="Reconnect to an existing run instead of starting one")
    parser.add_argument("--last-event-id", help="Reconnect after this SSE event ID")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events must be positive")
    if args.last_event_id is not None and args.run_id is None:
        parser.error("--last-event-id requires --run-id")
    if args.run_id is not None and (args.checkpoint or args.resumable):
        parser.error("--checkpoint and --resumable apply to new runs, not reconnects")
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
    if args.run_id is not None:
        run = {
            "thread_id": args.thread, "run_id": args.run_id,
            "last_event_id": args.last_event_id, "stream_mode": args.stream_mode,
        }
    else:
        options = {}
        if args.stream_mode:
            options["stream_mode"] = args.stream_mode
        if args.resumable:
            options["stream_resumable"] = True
        if args.checkpoint:
            options["checkpoint"] = {"checkpoint_id": args.checkpoint}
            run["input"] = None
        if options:
            run["run_options"] = options
    if args.use_async:
        asyncio.run(print_async(remote, run, args.max_events, args.run_id is not None))
    else:
        source = remote.reconnect(**run) if args.run_id is not None else remote.stream(**run)
        with closing(source) as events:
            for index, event in enumerate(events, start=1):
                print(event.model_dump_json(), flush=True)
                if args.max_events is not None and index >= args.max_events:
                    break


if __name__ == "__main__":
    main()
