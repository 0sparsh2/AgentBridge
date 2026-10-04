import asyncio

import pytest

from agentbridge_langchain.langfuse_api import LangfuseAPIClient
from agentbridge_langchain.langfuse_prompts import fetch_prompt


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
    with pytest.warns(DeprecationWarning, match="ingest_otlp"):
        client.ingest([{"id": "event-1", "type": "trace-create"}])
    client.create_score(trace_id="trace-1", name="quality", value=0.9, comment="good")
    assert '"batch": [{"id": "event-1", "type": "trace-create"}]' in calls[0][2].decode()
    assert '"traceId": "trace-1"' in calls[1][2].decode()


def test_langfuse_api_client_supports_async_json_and_sse_facades():
    def transport(method, url, headers, body):
        del url, headers, body
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok":true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"id":"trace-1"}\n\ndata: [DONE]\n'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    async def collect():
        response = await client.arequest_json("GET", "/api/public/health")
        events = [event async for event in client.astream_events("POST", "/api/public/ingestion")]
        otlp = await client.aingest_otlp({"resourceSpans": []})
        return response, events, otlp

    response, events, otlp = asyncio.run(collect())
    assert response == {"ok": True}
    assert events == [{"id": "trace-1"}]
    assert otlp.startswith(b"data:")


def test_langfuse_trace_and_dataset_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert client.get_trace("trace-1") == {"ok": True}
    assert client.create_dataset(name="refunds", description="regression") == {"ok": True}
    assert client.create_dataset_item(
        dataset_name="refunds", input={"text": "A123"}, expected_output={"eligible": True}
    ) == {"ok": True}
    assert calls[0][1].endswith("/api/public/traces/trace-1")
    assert b'"datasetName": "refunds"' in calls[-1][2]


def test_langfuse_dataset_v2_and_item_lifecycle_helpers_preserve_current_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    client.list_datasets(query={"limit": 10})
    client.get_dataset("refunds", version="2026-01-01T00:00:00Z")
    client.delete_dataset("refunds")
    client.create_dataset_item(
        dataset_name="refunds",
        input={"text": "A123"},
        expected_output={"eligible": True},
        metadata={"team": "support"},
        status="ACTIVE",
    )
    client.update_dataset_item("item-1", body={"datasetName": "refunds", "status": "ARCHIVED"})
    client.list_dataset_items(query={"datasetName": "refunds", "version": "2026-01-01"})
    client.get_dataset_item("item-1")
    client.delete_dataset_item("item-1")
    client.delete_trace("trace-1")
    client.delete_traces(trace_ids=["trace-2", "trace-3"])

    assert calls[0][1].endswith("/api/public/v2/datasets?limit=10")
    assert calls[1][1].endswith("/api/public/v2/datasets/refunds?version=2026-01-01T00%3A00%3A00Z")
    assert calls[3][0:2] == ("POST", "https://cloud.langfuse.com/api/public/dataset-items")
    assert calls[4][2] == b'{"datasetName": "refunds", "id": "item-1", "status": "ARCHIVED"}'
    assert calls[-1][1].endswith("/api/public/traces/delete")


def test_langfuse_current_telemetry_and_query_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if url.endswith("/otel/v1/traces"):
            assert headers["Content-Type"] == "application/x-protobuf"
            assert body == b"otlp-payload"
            return 200, {"content-type": "application/json"}, b'{"accepted":1}'
        return 200, {"content-type": "application/json"}, b'{"data":[]}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert client.ingest_otlp(b"otlp-payload", content_type="application/x-protobuf") == {
        "accepted": 1
    }
    assert client.list_observations(query={"limit": 10}) == {"data": []}
    assert client.list_scores_v3(query={"dataType": "NUMERIC"}) == {"data": []}
    assert client.query_metrics(query={"view": "traces"}) == {"data": []}
    assert client.list_experiments(query={"fields": "core,scores"}) == {"data": []}
    assert client.list_experiment_items(query={"experimentId": "exp-1", "fields": "io,scores"}) == {
        "data": []
    }
    assert calls[1][1].endswith("/api/public/v2/observations?limit=10")
    assert calls[2][1].endswith("/api/public/v3/scores?dataType=NUMERIC")
    assert calls[3][1].endswith("/api/public/v2/metrics?view=traces")
    assert calls[4][1].endswith("/api/public/experiments?fields=core%2Cscores")
    assert calls[5][1].endswith("/api/public/experiment-items?experimentId=exp-1&fields=io%2Cscores")


def test_langfuse_prompt_management_supports_versions_and_chat_compilation():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return (
            200,
            {"content-type": "application/json"},
            b'{"name":"refunds","type":"chat","version":3,"labels":["production"],'
            b'"prompt":[{"role":"system","content":"Help {{team}}"},'
            b'{"role":"user","content":"Order {{order_id}}"}]}',
        )

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    prompt = fetch_prompt(client, "refunds", label="production", prompt_type="chat")

    assert prompt.version == 3
    assert prompt.compile(team="support", order_id="A123")[1]["content"] == "Order A123"
    assert prompt.get_langchain_prompt()[0]["content"] == "Help {team}"
    assert calls[0][1].endswith("/api/public/v2/prompts/refunds?label=production&type=chat")
