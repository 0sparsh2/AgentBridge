import asyncio

import pytest

from agentbridge_langchain._sse import parse_sse
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
