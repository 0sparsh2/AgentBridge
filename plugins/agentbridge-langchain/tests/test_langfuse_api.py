from agentbridge_langchain.langfuse_api import LangfuseAPIClient


def test_langfuse_api_client_supports_json_and_sse_without_secrets_in_output():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok": true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"id": "trace-1"}\n\ndata: [DONE]\n'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert client.request_json("GET", "/api/public/health") == {"ok": True}
    assert list(client.stream_events("POST", "/api/public/ingestion", body={"batch": []})) == [{"id": "trace-1"}]
    assert calls[0][2]["Authorization"].startswith("Basic ")
    assert "pk" not in repr(client.request_json("GET", "/api/public/health"))


def test_langfuse_helpers_preserve_ingestion_and_score_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"id": "ok"}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    client.ingest([{"id": "event-1", "type": "trace-create"}])
    client.create_score(trace_id="trace-1", name="quality", value=0.9, comment="good")
    assert '"batch": [{"id": "event-1", "type": "trace-create"}]' in calls[0][2].decode()
    assert '"traceId": "trace-1"' in calls[1][2].decode()
