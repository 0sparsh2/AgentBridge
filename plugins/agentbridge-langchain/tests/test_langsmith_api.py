from __future__ import annotations

import json

from agentbridge_langchain.langsmith_api import LangSmithAPIClient


def test_langsmith_api_client_preserves_arbitrary_json_endpoint():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        return 200, {"content-type": "application/json"}, b'{"ok":true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    result = client.request_json(
        "POST",
        "/v1/threads",
        query={"limit": 2, "skip": None},
        body={"metadata": {"source": "agentbridge"}},
    )

    assert result == {"ok": True}
    assert calls[0][0:2] == ("POST", "https://example.test/v1/threads?limit=2")
    assert calls[0][2]["X-Api-Key"] == "secret"
    assert json.loads(calls[0][3]) == {"metadata": {"source": "agentbridge"}}


def test_langsmith_api_client_parses_sse_data_lines():
    def transport(method, url, headers, body):
        del method, url, headers, body
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)

    assert list(client.stream_events("POST", "v1/runs/stream")) == [{"event": "update"}]
