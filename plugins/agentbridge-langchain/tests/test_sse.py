import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentbridge_langchain._sse import async_events, parse_sse, response_events, stream_http
from agentbridge_langchain.langfuse_api import LangfuseAPIClient
from agentbridge_langchain.langsmith_api import LangSmithAPIClient


def test_sse_frames_preserve_names_multiline_data_and_persistent_resume_ids():
    response = (
        b'\xef\xbb\xbf: heartbeat\r\n'
        b'id: run-1-0\r\nevent: metadata\r\ndata: {"run_id":\r\ndata: "run-1"}\r\n\r\n'
        b'event: messages\ndata: [{"content":"hello"},{"node":"agent"}]\n\n'
        b'id:\nevent: end\ndata: null\n\n'
        b'data: [DONE]\n\ndata: {"ignored":true}\n\n'
    )
    assert list(parse_sse(response)) == [
        {"event": "metadata", "id": "run-1-0", "data": {"run_id": "run-1"}},
        {"event": "messages", "id": "run-1-0", "data": [
            {"content": "hello"}, {"node": "agent"}
        ]},
        {"event": "end", "id": "", "data": None},
    ]


def test_sse_data_only_envelopes_and_final_frame_remain_compatible():
    assert list(parse_sse(b'data: {"event":"update","data":{"x":1}}\n\n'
                          b'data: {"output":"done"}')) == [
        {"event": "update", "data": {"x": 1}}, {"output": "done"}
    ]
    assert list(parse_sse(b'id: invalid\x00id\ndata: {"x":1}\n\n')) == [{"x": 1}]


@pytest.mark.parametrize("provider", ["langsmith", "langfuse"])
def test_sync_and_async_transports_preserve_named_sse_frames(provider):
    def transport(method, url, headers, body):
        return 200, {"content-type": "text/event-stream"}, (
            b'event: error\nid: 2\ndata: {"message":"failed"}\n\n'
        )

    client = (
        LangSmithAPIClient(api_key="test", transport=transport)
        if provider == "langsmith" else
        LangfuseAPIClient(public_key="test", secret_key="test", transport=transport)
    )
    expected = [{"event": "error", "id": "2", "data": {"message": "failed"}}]
    assert list(client.stream_events("POST", "/stream")) == expected

    async def collect():
        return [event async for event in client.astream_events("POST", "/stream")]

    assert asyncio.run(collect()) == expected


def test_langfuse_stream_rejects_http_errors_before_parsing_events():
    def transport(method, url, headers, body):
        return 403, {}, b'permission denied'

    client = LangfuseAPIClient(public_key="test", secret_key="test", transport=transport)
    with pytest.raises(RuntimeError, match="Langfuse API 403"):
        list(client.stream_events("POST", "/stream"))


@pytest.fixture
def gated_sse_server():
    release = threading.Event()
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(b'event: messages\ndata: {"content":"first"}\n\n')
                self.wfile.flush()
                release.wait(5)
                self.wfile.write(b'event: end\ndata: {"output":"last"}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", release, finished
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _http_client(provider, base_url):
    if provider == "langsmith":
        return LangSmithAPIClient(api_key="test", base_url=base_url)
    return LangfuseAPIClient(public_key="test", secret_key="test", base_url=base_url)


@pytest.mark.parametrize("provider", ["langsmith", "langfuse"])
def test_http_stream_delivers_first_frame_before_response_finishes(provider, gated_sse_server):
    url, release, finished = gated_sse_server
    events = _http_client(provider, url).stream_events("GET", "/stream")
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first = pool.submit(next, events).result(timeout=2)
            assert first == {"event": "messages", "data": {"content": "first"}}
            assert not finished.is_set()
            release.set()
            assert list(events) == [{"event": "end", "data": {"output": "last"}}]
        finally:
            release.set()
            events.close()


@pytest.mark.parametrize("provider", ["langsmith", "langfuse"])
def test_async_http_stream_delivers_first_frame_without_waiting_for_eof(provider, gated_sse_server):
    url, release, finished = gated_sse_server

    async def collect():
        events = _http_client(provider, url).astream_events("GET", "/stream")
        try:
            first = await asyncio.wait_for(anext(events), timeout=2)
            assert first == {"event": "messages", "data": {"content": "first"}}
            assert not finished.is_set()
            release.set()
            assert [event async for event in events] == [
                {"event": "end", "data": {"output": "last"}}
            ]
        finally:
            release.set()
            await events.aclose()

    asyncio.run(collect())


@pytest.mark.parametrize("malformed", [False, True])
def test_stream_http_closes_response_on_early_close_or_decode_failure(monkeypatch, malformed):
    closed = []

    class Response:
        status = 200
        headers = {"content-type": "text/event-stream"}

        def __iter__(self):
            yield b'event: messages\n'
            yield b'data: invalid\n' if malformed else b'data: {"content":"first"}\n'
            yield b'\n'
            raise AssertionError("stream read ahead after first frame")

        def close(self):
            closed.append(True)

    monkeypatch.setattr("agentbridge_langchain._sse.urlopen", lambda *args, **kwargs: Response())
    _, _, lines = stream_http("GET", "https://server.test", {}, None)
    events = response_events(lines)
    if malformed:
        with pytest.raises(ValueError):
            next(events)
    else:
        assert next(events)["data"]["content"] == "first"
        events.close()
    assert closed == [True]


def test_async_stream_cancellation_waits_for_active_read_then_closes_iterator():
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def source():
        try:
            entered.set()
            assert release.wait(2)
            yield {"data": "event"}
        finally:
            closed.set()

    async def collect():
        events = async_events(source())
        pending = asyncio.create_task(anext(events))
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert closed.is_set()
        await events.aclose()

    try:
        asyncio.run(collect())
    finally:
        release.set()
