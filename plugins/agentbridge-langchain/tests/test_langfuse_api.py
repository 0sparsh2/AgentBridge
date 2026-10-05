import asyncio
import json

import pytest

from agentbridge_langchain.langfuse_api import LangfuseAPIClient
from agentbridge_langchain import afetch_prompt as exported_afetch_prompt
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


def test_langfuse_score_helper_preserves_typed_targets_and_native_options():
    calls = []

    def transport(method, url, headers, body):
        del method, url, headers
        calls.append(body)
        return 200, {"content-type": "application/json"}, b'{"id":"score-1"}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert client.create_score(
        score_id="score-1",
        name="helpfulness",
        value=True,
        session_id="session-1",
        observation_id="observation-1",
        data_type="BOOLEAN",
        config_id="config-1",
        metadata={"reviewer": "qa"},
        environment="staging",
    ) == {"id": "score-1"}

    assert json.loads(calls[0]) == {
        "id": "score-1",
        "name": "helpfulness",
        "value": True,
        "sessionId": "session-1",
        "observationId": "observation-1",
        "dataType": "BOOLEAN",
        "configId": "config-1",
        "metadata": {"reviewer": "qa"},
        "environment": "staging",
    }


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


def test_langfuse_async_typed_queries_preserve_current_paths_and_cursors():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "/observations" in url and "cursor=next" not in url:
            payload = {"data": [{"id": "observation-1"}], "meta": {"cursor": "next"}}
        elif "/observations" in url:
            payload = {"data": [{"id": "observation-2"}], "meta": {}}
        else:
            payload = {"data": []}
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    async def collect():
        observation = await client.alist_observations(query={"limit": 2})
        lookup = await client.aget_observation(
            "observation-1",
            from_start_time="2026-01-01T00:00:00Z",
            to_start_time="2026-01-02T00:00:00Z",
        )
        score = await client.alist_scores_v3(query={"dataType": "NUMERIC"})
        experiment = await client.alist_experiments(query={"limit": 2})
        items = await client.alist_dataset_items(query={"datasetName": "refunds"})
        observations = [item async for item in client.aiter_observations(query={"limit": 1})]
        return observation, lookup, score, experiment, items, observations

    observation, lookup, score, experiment, items, observations = asyncio.run(collect())
    assert observation["data"] == [{"id": "observation-1"}]
    assert lookup["data"] == [{"id": "observation-1"}]
    assert score == {"data": []}
    assert experiment == {"data": []}
    assert items == {"data": []}
    assert observations == [{"id": "observation-1"}, {"id": "observation-2"}]
    assert calls[0].endswith("/api/public/v2/observations?limit=2")
    assert calls[2].endswith("/api/public/v3/scores?dataType=NUMERIC")


def test_langfuse_async_cursor_iterators_preserve_filters_across_resources():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        resource = next(
            name
            for name in ("experiments", "experiment-items", "datasets", "dataset-items")
            if f"/{name}" in url
        )
        if "cursor=next" not in url:
            payload = {"data": [{"resource": resource, "page": 1}], "meta": {"cursor": "next"}}
        else:
            payload = {"data": [{"resource": resource, "page": 2}], "meta": {}}
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    async def collect():
        experiments = [item async for item in client.aiter_experiments(query={"limit": 1})]
        experiment_items = [
            item async for item in client.aiter_experiment_items(query={"experimentId": "exp-1"})
        ]
        datasets = [item async for item in client.aiter_datasets(query={"limit": 1})]
        dataset_items = [
            item async for item in client.aiter_dataset_items(query={"datasetName": "refunds"})
        ]
        return experiments, experiment_items, datasets, dataset_items

    values = asyncio.run(collect())
    assert all(len(items) == 2 for items in values)
    assert values[0][0]["resource"] == "experiments"
    assert values[1][0]["resource"] == "experiment-items"
    assert values[2][0]["resource"] == "datasets"
    assert values[3][0]["resource"] == "dataset-items"
    assert calls[0].endswith("/api/public/experiments?limit=1")
    assert calls[1].endswith("/api/public/experiments?limit=1&cursor=next")
    assert calls[2].endswith("/api/public/experiment-items?experimentId=exp-1")


def test_langfuse_async_lifecycle_helpers_preserve_payloads_and_current_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    async def collect():
        prompt = await client.aget_prompt("refunds", label="production", prompt_type="text")
        created_prompt = await client.acreate_prompt(
            name="refunds",
            prompt="Handle {input}",
            prompt_type="text",
            labels=["production"],
        )
        score = await client.acreate_score(
            name="quality",
            value=True,
            trace_id="trace-1",
            data_type="BOOLEAN",
        )
        dataset = await client.acreate_dataset(name="refunds", description="regression")
        fetched_dataset = await client.aget_dataset("refunds", version="v1")
        item = await client.acreate_dataset_item(
            dataset_name="refunds",
            input={"question": "double charge"},
            expected_output={"eligible": True},
            status="ACTIVE",
        )
        return prompt, created_prompt, score, dataset, fetched_dataset, item

    assert asyncio.run(collect()) == ({"ok": True},) * 6
    assert calls[0][1].endswith("/api/public/v2/prompts/refunds?label=production&type=text")
    assert json.loads(calls[1][2])["labels"] == ["production"]
    assert json.loads(calls[2][2]) == {
        "name": "quality",
        "value": True,
        "traceId": "trace-1",
        "dataType": "BOOLEAN",
    }
    assert calls[5][1].endswith("/api/public/dataset-items")


def test_langfuse_async_metrics_prompts_and_cleanup_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if "/prompts" in url and "cursor=next" not in url:
            payload = {"data": [{"name": "refunds"}], "meta": {"cursor": "next"}}
        elif "/prompts" in url:
            payload = {"data": [{"name": "support"}], "meta": {}}
        else:
            payload = {"ok": True}
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else json.dumps(payload).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    async def collect():
        metrics = await client.aquery_metrics(query={"view": "traces"})
        prompts = [item async for item in client.aiter_prompts(query={"limit": 1})]
        trace = await client.aget_trace("trace-1")
        await client.adelete_trace("trace-1")
        await client.adelete_traces(trace_ids=["trace-1", "trace-2"])
        item = await client.aget_dataset_item("item-1")
        await client.adelete_dataset_item("item-1")
        return metrics, prompts, trace, item

    metrics, prompts, trace, item = asyncio.run(collect())
    assert metrics == {"ok": True}
    assert prompts == [{"name": "refunds"}, {"name": "support"}]
    assert trace == {"ok": True}
    assert item == {"ok": True}
    assert calls[0][1].endswith("/api/public/v2/metrics?view=traces")
    assert calls[1][1].endswith("/api/public/v2/prompts?limit=1")
    assert calls[2][1].endswith("/api/public/v2/prompts?limit=1&cursor=next")
    assert calls[-1][1].endswith("/api/public/dataset-items/item-1")


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


def test_langfuse_dataset_item_iterator_preserves_filters_across_cursors():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "cursor=next" in url:
            payload = {"data": [{"id": "item-2"}], "meta": {}}
        else:
            payload = {"data": [{"id": "item-1"}], "meta": {"cursor": "next"}}
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    assert list(client.iter_dataset_items(query={"datasetName": "refunds", "version": "v1"})) == [
        {"id": "item-1"},
        {"id": "item-2"},
    ]
    assert calls == [
        "https://cloud.langfuse.com/api/public/dataset-items?datasetName=refunds&version=v1",
        "https://cloud.langfuse.com/api/public/dataset-items?datasetName=refunds&version=v1&cursor=next",
    ]


def test_langfuse_dataset_and_prompt_iterators_preserve_filters_across_cursors():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        kind = "prompts" if "/prompts" in url else "datasets"
        suffix = "-2" if "cursor=next" in url else "-1"
        payload = {"data": [{"id": f"{kind}{suffix}"}], "meta": {}}
        if "cursor=next" not in url:
            payload["meta"] = {"cursor": "next"}
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    assert list(client.iter_datasets(query={"limit": 2})) == [
        {"id": "datasets-1"},
        {"id": "datasets-2"},
    ]
    assert list(client.iter_prompts(query={"label": "production", "type": "text"})) == [
        {"id": "prompts-1"},
        {"id": "prompts-2"},
    ]
    assert calls == [
        "https://cloud.langfuse.com/api/public/v2/datasets?limit=2",
        "https://cloud.langfuse.com/api/public/v2/datasets?limit=2&cursor=next",
        "https://cloud.langfuse.com/api/public/v2/prompts?label=production&type=text",
        "https://cloud.langfuse.com/api/public/v2/prompts?label=production&type=text&cursor=next",
    ]


def test_langfuse_current_telemetry_and_query_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if url.endswith("/otel/v1/traces"):
            assert headers["Content-Type"] == "application/x-protobuf"
            assert headers["x-langfuse-ingestion-version"] == "4"
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


def test_langfuse_current_observation_and_score_lookup_helpers_use_v4_query_contracts():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        return 200, {"content-type": "application/json"}, b'{"data": []}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    client.list_trace_observations(
        "trace-1",
        from_start_time="2026-01-01T00:00:00Z",
        to_start_time="2026-01-02T00:00:00Z",
        fields="core,usage",
    )
    client.get_observation(
        "observation-1",
        from_start_time="2026-01-01T00:00:00Z",
        to_start_time="2026-01-02T00:00:00Z",
        fields="core,io",
    )
    client.get_score("score-1", fields="details,subject")

    assert "traceId=trace-1" in calls[0]
    assert "fromStartTime=2026-01-01T00%3A00%3A00Z" in calls[0]
    assert "fields=core%2Cusage" in calls[0]
    assert "filter=%5B%7B%22type%22%3A%22string%22" in calls[1]
    assert "observation-1" in calls[1]
    assert calls[2].endswith("/api/public/v3/scores?id=score-1&fields=details%2Csubject")


def test_langfuse_query_iterators_follow_cursor_pages_without_dropping_filters():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "/observations" in url:
            if "cursor=next" in url:
                payload = b'{"data":[{"id":"obs-2"}],"meta":{"cursor":null}}'
            else:
                payload = b'{"data":[{"id":"obs-1"}],"meta":{"cursor":"next"}}'
        else:
            if "cursor=score-next" in url:
                payload = b'{"data":[{"id":"score-2"}],"meta":{"cursor":null}}'
            else:
                payload = b'{"data":[{"id":"score-1"}],"meta":{"cursor":"score-next"}}'
        return 200, {"content-type": "application/json"}, payload

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)

    assert [item["id"] for item in client.iter_observations(query={"traceId": "trace-1"})] == [
        "obs-1",
        "obs-2",
    ]
    assert [item["id"] for item in client.iter_scores_v3(query={"dataType": "NUMERIC"})] == [
        "score-1",
        "score-2",
    ]
    assert "traceId=trace-1" in calls[0]
    assert "cursor=next" in calls[1]
    assert "dataType=NUMERIC" in calls[2]
    assert "cursor=score-next" in calls[3]


def test_langfuse_experiment_iterators_follow_cursor_pages_without_dropping_filters():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "/experiments?" in url:
            if "cursor=next" in url:
                payload = b'{"data":[{"id":"exp-2"}],"meta":{"cursor":null}}'
            else:
                payload = b'{"data":[{"id":"exp-1"}],"meta":{"cursor":"next"}}'
        elif "cursor=item-next" in url:
            payload = b'{"data":[{"id":"item-2"}],"meta":{"cursor":null}}'
        else:
            payload = b'{"data":[{"id":"item-1"}],"meta":{"cursor":"item-next"}}'
        return 200, {"content-type": "application/json"}, payload

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert [item["id"] for item in client.iter_experiments(query={"fromStartTime": "2026-01-01"})] == [
        "exp-1",
        "exp-2",
    ]
    assert [item["id"] for item in client.iter_experiment_items(query={"experimentId": "exp-1"})] == [
        "item-1",
        "item-2",
    ]
    assert "fromStartTime=2026-01-01" in calls[0]
    assert "cursor=next" in calls[1]
    assert "experimentId=exp-1" in calls[2]
    assert "cursor=item-next" in calls[3]


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
    async_prompt = asyncio.run(
        exported_afetch_prompt(client, "refunds", label="production", prompt_type="chat")
    )

    assert prompt.version == 3
    assert async_prompt.version == 3
    assert async_prompt.compile(team="support", order_id="A123")[1]["content"] == "Order A123"
    assert prompt.compile(team="support", order_id="A123")[1]["content"] == "Order A123"
    assert prompt.get_langchain_prompt()[0]["content"] == "Help {team}"
    assert calls[0][1].endswith("/api/public/v2/prompts/refunds?label=production&type=chat")
