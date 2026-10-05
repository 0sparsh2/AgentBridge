from __future__ import annotations

import json
import asyncio

from agentbridge_langchain.langsmith_api import LangSmithAPIClient, LangSmithControlPlaneClient


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
        del url, headers, body
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok":true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)

    assert list(client.stream_events("POST", "v1/runs/stream")) == [{"event": "update"}]


def test_langsmith_universal_call_covers_json_sse_and_async_endpoints():
    def transport(method, url, headers, body):
        del method, url, headers, body
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert list(client.call("POST", "/custom/stream", stream=True)) == [{"event": "update"}]

    async def collect():
        events = await client.acall("POST", "/custom/stream", stream=True)
        return [event async for event in events]

    assert asyncio.run(collect()) == [{"event": "update"}]


def test_langsmith_api_client_supports_async_json_and_sse_facades():
    def transport(method, url, headers, body):
        del url, headers, body
        if method == "GET":
            return 200, {"content-type": "application/json"}, b'{"ok":true}'
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)

    async def collect():
        response = await client.arequest_json("GET", "/v1/threads")
        events = [event async for event in client.astream_events("POST", "/v1/runs/stream")]
        return response, events

    response, events = asyncio.run(collect())
    assert response == {"ok": True}
    assert events == [{"event": "update"}]


def test_langsmith_dataset_delta_and_shared_examples_are_typed_sync_and_async():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(
        api_key="secret",
        base_url="https://example.test",
        transport=transport,
    )
    expected_delta = {
        "baseline_session_id": "baseline",
        "comparison_session_ids": ["candidate"],
        "feedback_key": "quality",
        "filters": {"tag": "refunds"},
        "offset": 2,
        "limit": 10,
        "comparative_experiment_id": "experiment-1",
    }
    client.read_dataset_delta(
        "dataset-1",
        baseline_session_id="baseline",
        comparison_session_ids=["candidate"],
        feedback_key="quality",
        filters={"tag": "refunds"},
        offset=2,
        limit=10,
        comparative_experiment_id="experiment-1",
    )
    client.read_shared_dataset_examples_with_runs(
        "share-1", body={"dataset_id": "dataset-1", "limit": 5}
    )

    async def collect():
        return (
            await client.aread_dataset_delta(
                "dataset-1",
                baseline_session_id="baseline",
                comparison_session_ids=["candidate"],
                feedback_key="quality",
                filters={"tag": "refunds"},
                offset=2,
                limit=10,
                comparative_experiment_id="experiment-1",
            ),
            await client.aread_shared_dataset_examples_with_runs(
                "share-1", body={"dataset_id": "dataset-1", "limit": 5}
            ),
        )

    assert asyncio.run(collect()) == ({"ok": True}, {"ok": True})
    assert calls[0] == (
        "POST",
        "https://example.test/api/v1/datasets/dataset-1/runs/delta",
        expected_delta,
    )
    assert calls[1][1].endswith("/api/v1/public/share-1/examples/runs")
    assert calls[2] == calls[0]
    assert calls[3] == calls[1]


def test_langsmith_connection_token_lifecycle_preserves_sync_and_async_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(
        api_key="secret",
        base_url="https://example.test",
        transport=transport,
    )
    client.list_connection_tokens(query={"agent_id": "agent-1"})
    client.update_connection_token(
        "token-1", body={"label": "production", "is_default": True}
    )
    assert client.revoke_connection_token("token-1") is None

    async def collect():
        return (
            await client.alist_connection_tokens(query={"agent_id": "agent-1"}),
            await client.aupdate_connection_token(
                "token-1", body={"label": "production", "is_default": True}
            ),
            await client.arevoke_connection_token("token-1"),
        )

    assert asyncio.run(collect()) == ({"ok": True}, {"ok": True}, None)
    assert calls[0][1].endswith("/v1/fleet/auth-tokens?agent_id=agent-1")
    assert calls[1] == (
        "PATCH",
        "https://example.test/v1/fleet/auth-tokens/token-1",
        {"label": "production", "is_default": True},
    )
    assert calls[2][0:2] == (
        "DELETE",
        "https://example.test/v1/fleet/auth-tokens/token-1",
    )


def test_langsmith_dataset_versions_splits_and_sharing_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.read_dataset_version("dataset-1", tag="production")
    client.list_dataset_versions("dataset-1", query={"limit": 5})
    client.diff_dataset_versions("dataset-1", from_version="dev", to_version="production")
    client.list_dataset_splits("dataset-1", as_of="production")
    client.update_dataset_splits(
        "dataset-1", split_name="holdout", example_ids=["example-1"], remove=True
    )
    client.share_dataset("dataset-1")
    assert client.unshare_dataset("dataset-1") is None

    async def collect():
        return (
            await client.aread_dataset_version("dataset-1", as_of="2026-01-01T00:00:00Z"),
            await client.alist_dataset_versions("dataset-1", query={"limit": 5}),
            await client.adiff_dataset_versions(
                "dataset-1", from_version="dev", to_version="production"
            ),
            await client.alist_dataset_splits("dataset-1", as_of="production"),
            await client.aupdate_dataset_splits(
                "dataset-1", split_name="holdout", example_ids=["example-1"], remove=True
            ),
            await client.ashare_dataset("dataset-1"),
            await client.aunshare_dataset("dataset-1"),
        )

    assert asyncio.run(collect()) == (
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        {"ok": True},
        None,
    )
    assert calls[0][1].endswith("/api/v1/datasets/dataset-1/version?tag=production")
    assert calls[1][1].endswith("/api/v1/datasets/dataset-1/versions?limit=5")
    assert calls[2][1].endswith(
        "/api/v1/datasets/dataset-1/versions/diff?from_version=dev&to_version=production"
    )
    assert calls[4][2] == {
        "split_name": "holdout",
        "examples": ["example-1"],
        "remove": True,
    }
    assert calls[5][2] == {"dataset_id": "dataset-1"}
    assert calls[6][0:2] == ("DELETE", "https://example.test/api/v1/datasets/dataset-1/share")


def test_langsmith_bulk_examples_shared_reads_and_openai_export_preserve_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.delete_examples(["example-1", "example-2"]) is None
    client.delete_examples(["example-1"], hard_delete=True)
    client.list_shared_examples("share-1", example_ids=["example-1"], limit=2)
    client.read_dataset_openai_finetuning("dataset-1")

    async def collect():
        return (
            await client.adelete_examples(["example-1", "example-2"]),
            await client.adelete_examples(["example-1"], hard_delete=True),
            await client.alist_shared_examples("share-1", example_ids=["example-1"], limit=2),
            await client.aread_dataset_openai_finetuning("dataset-1"),
        )

    assert asyncio.run(collect()) == (None, {"ok": True}, {"ok": True}, {"ok": True})
    assert calls[0][1].endswith(
        "/api/v1/examples?example_ids=example-1&example_ids=example-2"
    )
    assert calls[1] == (
        "POST",
        "https://example.test/api/v1/platform/datasets/examples/delete",
        {"example_ids": ["example-1"], "hard_delete": True},
    )
    assert calls[2][1].endswith("/api/v1/public/share-1/examples?id=example-1&limit=2")
    assert calls[3][1].endswith("/api/v1/datasets/dataset-1/openai_ft")


def test_langsmith_annotation_queue_lifecycle_preserves_sync_and_async_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.create_annotation_queue(
        name="Refund review", description="Human review", rubric_instructions="Check eligibility"
    )
    client.list_annotation_queues(query={"name_contains": "Refund"})
    client.get_annotation_queue("queue-1")
    client.update_annotation_queue("queue-1", body={"description": "Updated"})
    client.add_runs_to_annotation_queue("queue-1", run_ids=["run-1"])
    client.add_runs_to_annotation_queue(
        "queue-1", runs=[{"run_id": "run-2", "session_id": "session-1"}]
    )
    client.list_annotation_queue_runs("queue-1", query={"status": "needs_my_review"})
    client.get_annotation_queue_run("queue-1", 0)
    assert client.remove_run_from_annotation_queue("queue-1", "run-1") is None
    assert client.delete_annotation_queue("queue-1") is None

    async def collect():
        return [
            await client.acreate_annotation_queue(name="Refund review"),
            await client.alist_annotation_queues(query={"name_contains": "Refund"}),
            await client.aget_annotation_queue("queue-1"),
            await client.aupdate_annotation_queue("queue-1", body={"description": "Updated"}),
            await client.aadd_runs_to_annotation_queue("queue-1", run_ids=["run-1"]),
            await client.aadd_runs_to_annotation_queue(
                "queue-1", runs=[{"run_id": "run-2", "session_id": "session-1"}]
            ),
            await client.alist_annotation_queue_runs(
                "queue-1", query={"status": "needs_my_review"}
            ),
            await client.aget_annotation_queue_run("queue-1", 0),
            await client.aremove_run_from_annotation_queue("queue-1", "run-1"),
            await client.adelete_annotation_queue("queue-1"),
        ]

    values = asyncio.run(collect())
    assert values[:8] == [{"ok": True}] * 8
    assert values[8:] == [None, None]
    assert calls[0][2] == {
        "name": "Refund review",
        "description": "Human review",
        "rubric_instructions": "Check eligibility",
    }
    assert calls[4] == (
        "POST",
        "https://example.test/api/v1/annotation-queues/queue-1/runs",
        ["run-1"],
    )
    assert calls[5][1].endswith("/api/v1/annotation-queues/queue-1/runs/by-key")


def test_langsmith_feedback_config_and_presigned_token_lifecycle_preserves_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, json.loads(body) if body else None))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.create_feedback_config(
        "quality", feedback_config={"type": "continuous", "min": 0, "max": 1}
    )
    client.list_feedback_configs(query={"name_contains": "qual"})
    client.update_feedback_config("quality", body={"is_lower_score_better": True})
    assert client.delete_feedback_config("quality") is None
    client.create_presigned_feedback_token(
        "run-1", "quality", body={"expires_in": {"hours": 1}}
    )
    client.list_presigned_feedback_tokens("run-1", query={"limit": 5})

    async def collect():
        return (
            await client.acreate_feedback_config(
                "quality", feedback_config={"type": "continuous", "min": 0, "max": 1}
            ),
            await client.alist_feedback_configs(query={"name_contains": "qual"}),
            await client.aupdate_feedback_config(
                "quality", body={"is_lower_score_better": True}
            ),
            await client.adelete_feedback_config("quality"),
            await client.acreate_presigned_feedback_token(
                "run-1", "quality", body={"expires_in": {"hours": 1}}
            ),
            await client.alist_presigned_feedback_tokens("run-1", query={"limit": 5}),
        )

    assert asyncio.run(collect()) == (
        {"ok": True},
        {"ok": True},
        {"ok": True},
        None,
        {"ok": True},
        {"ok": True},
    )
    assert calls[0][2]["feedback_key"] == "quality"
    assert calls[1][1].endswith("/api/v1/feedback-configs?name_contains=qual")
    assert calls[3][1].endswith("/api/v1/feedback-configs?feedback_key=quality")
    assert calls[4][2] == {
        "run_id": "run-1",
        "feedback_key": "quality",
        "expires_in": {"hours": 1},
    }
    assert calls[5][1].endswith("/api/v1/feedback/tokens?run_id=run-1&limit=5")


def test_langsmith_async_typed_lifecycle_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, body))
        if url.endswith("/runs/stream"):
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"token"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        thread = await client.acreate_thread(metadata={"team": "support"})
        fetched = await client.aget_thread("thread-1")
        searched = await client.asearch_threads(body={"limit": 2})
        state = await client.aget_thread_state("thread-1", checkpoint_id="cp-1")
        updated = await client.aupdate_thread_state("thread-1", values={"approved": True})
        run = await client.acreate_thread_run(
            "thread-1", assistant_id="assistant-1", input={"text": "hello"}
        )
        stream = await client.acreate_thread_run(
            "thread-1", assistant_id="assistant-1", input={"text": "stream"}, stream=True
        )
        events = [event async for event in stream]
        waited = await client.acreate_run_wait(body={"assistant_id": "assistant-1"})
        background = await client.acreate_background_run(body={"assistant_id": "assistant-1"})
        cancelled = await client.acancel_run("thread-1", "run-1")
        feedback = await client.acreate_feedback(run_id="run-1", key="quality", score=1)
        return thread, fetched, searched, state, updated, run, events, waited, background, cancelled, feedback

    values = asyncio.run(collect())
    assert all(value == {"ok": True} for value in values if not isinstance(value, list))
    assert values[6] == [{"event": "token"}]
    assert calls[0][1].endswith("/threads")
    assert calls[3][1].endswith("/threads/thread-1/state?checkpoint_id=cp-1")
    assert calls[-1][1].endswith("/api/v1/feedback")


def test_langsmith_async_thread_checkpoint_and_system_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if headers.get("Accept") == "text/event-stream":
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"run"}\n\ndata: [DONE]\n'
        if url.endswith("/metrics?format=prometheus"):
            return 200, {"content-type": "text/plain"}, b"agent_runs_total 1\n"
        if url.endswith("/docs"):
            return 200, {"content-type": "text/html"}, b"<html>docs</html>"
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        history = await client.athread_history("thread-1", body={"limit": 1})
        history_get = await client.aget_thread_history("thread-1", limit=2)
        stream = [
            event
            async for event in client.ajoin_thread_stream(
                "thread-1", stream_modes=["lifecycle"], last_event_id="7"
            )
        ]
        resolved = await client.aresolve_interrupt("thread-1")
        pruned = await client.aprune_threads(thread_ids=["thread-1"], strategy="keep_latest")
        checkpoint = await client.aget_thread_state_at_checkpoint("thread-1", "checkpoint-1", subgraphs=True)
        checkpoint_body = await client.aget_thread_state_at_checkpoint_body(
            "thread-1", body={"checkpoint": {"checkpoint_id": "checkpoint-1"}}
        )
        searched = await client.asearch_runs(body={"assistant_id": "agent"})
        health = await client.ahealth_check(check_db=True)
        info = await client.aserver_info()
        docs = await client.aapi_documentation()
        metrics = await client.asystem_metrics()
        return history, history_get, stream, resolved, pruned, checkpoint, checkpoint_body, searched, health, info, docs, metrics

    values = asyncio.run(collect())
    assert all(value == {"ok": True} for value in values[:2] + values[3:10])
    assert values[2] == [{"event": "run"}]
    assert values[10] == b"<html>docs</html>"
    assert values[11] == b"agent_runs_total 1\n"
    assert calls[0][1].endswith("/threads/thread-1/history")
    assert calls[1][1].endswith("/threads/thread-1/history?limit=2")
    assert calls[2][1].endswith("/threads/thread-1/stream?stream_modes=lifecycle")
    assert calls[2][2]["Last-Event-ID"] == "7"
    assert any(url.endswith("/threads/thread-1/state/checkpoint-1?subgraphs=True") for _, url, _, _ in calls)
    assert any(url.endswith("/ok?check_db=1") for _, url, _, _ in calls)


def test_langsmith_async_assistant_and_run_lifecycle_preserves_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, body))
        if url.endswith("/join") and headers.get("Accept") == "text/event-stream":
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"done"}\n\ndata: [DONE]\n'
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        created = await client.acreate_assistant(graph_id="agent", config={"model": "offline"})
        fetched = await client.aget_assistant("assistant-1")
        await client.aupdate_assistant("assistant-1", config={"prompt": "hello"})
        await client.asearch_assistants(body={"limit": 2})
        await client.acount_assistants(body={"graph_id": "agent"})
        await client.aget_assistant_graph("assistant-1", xray=2)
        await client.aget_assistant_schemas("assistant-1")
        await client.aget_assistant_subgraphs("assistant-1", namespace="child/ns")
        await client.aget_assistant_versions("assistant-1", query={"limit": 10})
        await client.aset_latest_assistant_version("assistant-1", 3)
        await client.adelete_assistant("assistant-1")
        await client.acount_threads(body={"metadata": {"team": "support"}})
        await client.apatch_thread("thread-1", body={"metadata": {"team": "support"}})
        await client.acopy_thread("thread-1")
        run = await client.aget_run("thread-1", "run-1")
        runs = await client.alist_thread_runs("thread-1", query={"limit": 10})
        events = await client.alist_run_events("thread-1", "run-1")
        joined = await client.ajoin_run("thread-1", "run-1")
        stream = [event async for event in client.ajoin_run_stream("thread-1", "run-1")]
        await client.acancel_runs("thread-1", body={"run_ids": ["run-1"]})
        return created, fetched, run, runs, events, joined, stream

    values = asyncio.run(collect())
    assert all(value == {"ok": True} for value in values[:-1])
    assert values[-1] == [{"event": "done"}]
    assert calls[0][1].endswith("/assistants")
    assert calls[5][1].endswith("/assistants/assistant-1/graph?xray=2")
    assert calls[7][1].endswith("/assistants/assistant-1/subgraphs/child%2Fns")
    assert any(url.endswith("/threads/thread-1/runs/run-1/events") for _, url, _ in calls)
    assert calls[-1][1].endswith("/threads/thread-1/runs/cancel")


def test_langsmith_async_dataset_example_and_feedback_helpers_preserve_cursors():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if "/datasets?" in url:
            payload = (
                {"datasets": [{"id": "dataset-2"}]}
                if "cursor=next" in url
                else {"datasets": [{"id": "dataset-1"}], "next_cursor": "next"}
            )
        elif "/examples?" in url:
            payload = (
                {"examples": [{"id": "example-2"}]}
                if "cursor=next" in url
                else {"examples": [{"id": "example-1"}], "next_cursor": "next"}
            )
        else:
            payload = {"ok": True}
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else json.dumps(payload).encode()

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        dataset = await client.acreate_dataset(name="refunds", metadata={"team": "support"})
        datasets = [item async for item in client.aiter_datasets(query={"name": "refunds"})]
        await client.aupdate_dataset("dataset-1", body={"description": "Updated"})
        await client.adelete_dataset("dataset-1")
        example = await client.acreate_example(
            dataset_id="dataset-1",
            inputs={"question": "double charged"},
            outputs={"eligible": True},
            example_id="example-1",
        )
        examples = [item async for item in client.aiter_examples(query={"dataset_id": "dataset-1"})]
        await client.aupdate_example("example-1", body={"metadata": {"reviewed": True}})
        await client.adelete_example("example-1")
        feedback = await client.acreate_feedback(
            run_id="run-1", key="quality", score=1, comment="Helpful"
        )
        listed = await client.alist_feedback(query={"run_id": "run-1"})
        fetched = await client.aget_feedback("feedback-1")
        await client.aupdate_feedback("feedback-1", body={"comment": "Updated"})
        await client.adelete_feedback("feedback-1")
        return dataset, datasets, example, examples, feedback, listed, fetched

    values = asyncio.run(collect())
    assert values[0] == {"ok": True}
    assert [item["id"] for item in values[1]] == ["dataset-1", "dataset-2"]
    assert values[2] == {"ok": True}
    assert [item["id"] for item in values[3]] == ["example-1", "example-2"]
    assert values[4:] == ({"ok": True},) * 3
    assert calls[0][1].endswith("/api/v1/datasets")
    assert calls[1][1].endswith("/api/v1/datasets?name=refunds")
    assert calls[2][1].endswith("/api/v1/datasets?name=refunds&cursor=next")
    assert calls[6][1].endswith("/api/v1/examples?dataset_id=dataset-1")
    assert calls[7][1].endswith("/api/v1/examples?dataset_id=dataset-1&cursor=next")


def test_langsmith_async_a2a_mcp_and_cron_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if "/a2a/" in url and headers.get("Accept") == "text/event-stream":
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"task"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        a2a = await client.aa2a_json_rpc("assistant-1", body={"method": "message/send"})
        a2a_stream = [
            event async for event in client.aa2a_stream("assistant-1", body={"method": "message/stream"})
        ]
        mcp_get = await client.amcp_get()
        mcp_post = await client.amcp_post(body={"method": "tools/list"})
        mcp_delete = await client.amcp_terminate()
        cron = await client.acreate_cron(body={"schedule": "@hourly"})
        thread_cron = await client.acreate_thread_cron("thread-1", body={"schedule": "@daily"})
        searched = await client.asearch_crons(body={"assistant_id": "assistant-1"})
        counted = await client.acount_crons()
        fetched = await client.aget_cron("cron-1")
        updated = await client.aupdate_cron("cron-1", body={"enabled": False})
        deleted = await client.adelete_cron("cron-1")
        return a2a, a2a_stream, mcp_get, mcp_post, mcp_delete, cron, thread_cron, searched, counted, fetched, updated, deleted

    values = asyncio.run(collect())
    assert all(value == {"ok": True} for value in values if not isinstance(value, list))
    assert values[1] == [{"event": "task"}]
    assert calls[0][1].endswith("/a2a/assistant-1")
    assert calls[0][2]["Accept"] == "application/json"
    assert calls[2][1].endswith("/mcp/")
    assert calls[3][2]["Accept"] == "application/json"
    assert any(url.endswith("/runs/crons/search") for _, url, _, _ in calls)
    assert calls[-1][1].endswith("/runs/crons/cron-1")


def test_langsmith_thread_helpers_preserve_native_api_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/runs/stream"):
            return 200, {}, b'data: {"event": "message"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"id": "thread-1"}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.create_thread(metadata={"team": "support"})["id"] == "thread-1"
    assert client.get_thread("thread-1")["id"] == "thread-1"
    assert list(client.stream_thread_run("thread-1", assistant_id="assistant", input={"x": 1})) == [
        {"event": "message"}
    ]
    assert calls[0][2] == b'{"metadata": {"team": "support"}}'


def test_langsmith_assistant_run_and_state_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert client.create_assistant(graph_id="agent", config={"model": "offline"}) == {"ok": True}
    assert client.update_assistant("assistant-1", config={"prompt": "hello"}) == {"ok": True}
    assert client.create_thread_run("thread-1", assistant_id="assistant-1", input={"x": 1}) == {"ok": True}
    assert client.get_thread_state("thread-1", checkpoint_id="cp-1") == {"ok": True}
    assert client.update_thread_state("thread-1", values={"approved": True}, as_node="review") == {"ok": True}
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/assistants")
    assert calls[-1][2] == b'{"values": {"approved": true}, "as_node": "review"}'


def test_langsmith_assistant_introspection_and_thread_run_lifecycle_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/join"):
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"done"}\n\ndata: [DONE]\n'
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.count_assistants(body={"graph_id": "agent"})
    client.get_assistant_graph("assistant-1", xray=2)
    client.get_assistant_schemas("assistant-1")
    client.get_assistant_subgraphs("assistant-1")
    client.get_assistant_subgraphs("assistant-1", namespace="child/ns")
    client.get_assistant_versions("assistant-1", query={"limit": 10})
    client.patch_thread("thread-1", body={"metadata": {"team": "support"}})
    client.copy_thread("thread-1")
    client.list_thread_runs("thread-1", query={"limit": 10})
    client.list_run_events("thread-1", "run-1")
    client.join_run("thread-1", "run-1")
    assert list(client.join_run_stream("thread-1", "run-1")) == [{"event": "done"}]
    client.cancel_runs("thread-1", body={"run_ids": ["run-1"]})

    assert calls[0][1].endswith("/assistants/count")
    assert calls[1][1].endswith("/assistants/assistant-1/graph?xray=2")
    assert calls[4][1].endswith("/assistants/assistant-1/subgraphs/child%2Fns")
    assert calls[7][1].endswith("/threads/thread-1/copy")
    assert calls[8][1].endswith("/threads/thread-1/runs?limit=10")
    assert calls[10][1].endswith("/threads/thread-1/runs/run-1/join")


def test_langsmith_stateless_runs_and_cron_lifecycle_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if url.endswith("/runs/stream"):
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"update"}\n\ndata: [DONE]\n'
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    body = {"assistant_id": "agent", "input": {"messages": []}, "context": {"tenant": "support"}}
    client.create_background_run(body=body)
    client.create_run_wait(body=body)
    assert list(client.create_run_stream(body=body)) == [{"event": "update"}]
    client.create_run_batch(body={"runs": [body]})
    client.create_cron(body={"schedule": "0 * * * *", **body})
    client.create_thread_cron("thread-1", body={"schedule": "@hourly", **body})
    client.search_crons(body={"assistant_id": "agent"})
    client.count_crons()
    client.get_cron("cron-1")
    client.update_cron("cron-1", body={"enabled": False})
    assert client.delete_cron("cron-1") is None

    assert calls[0][1].endswith("/runs")
    assert calls[1][1].endswith("/runs/wait")
    assert calls[2][1].endswith("/runs/stream")
    assert calls[4][1].endswith("/runs/crons")
    assert calls[5][1].endswith("/threads/thread-1/runs/crons")
    assert calls[-1][1].endswith("/runs/crons/cron-1")


def test_langsmith_protocol_helpers_preserve_a2a_and_mcp_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if method == "DELETE":
            return 204, {}, b""
        return 200, {"content-type": "application/json"}, b'{"jsonrpc":"2.0","result":{}}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {}}}
    assert client.a2a_json_rpc("assistant-1", body=rpc) == {"jsonrpc": "2.0", "result": {}}
    assert client.mcp_get() == {"jsonrpc": "2.0", "result": {}}
    assert client.mcp_post(body={"jsonrpc": "2.0", "id": 2, "method": "initialize"}) == {
        "jsonrpc": "2.0",
        "result": {},
    }
    assert client.mcp_terminate() is None

    assert calls[0][0:2] == ("POST", "https://example.test/a2a/assistant-1")
    assert calls[0][2]["Accept"] == "application/json"
    assert json.loads(calls[0][3]) == rpc
    assert calls[1][0:2] == ("GET", "https://example.test/mcp/")
    assert calls[2][2]["Accept"] == "application/json"
    assert calls[3][0:2] == ("DELETE", "https://example.test/mcp/")


def test_langsmith_protocol_helpers_stream_a2a_json_rpc_events():
    def transport(method, url, headers, body):
        assert method == "POST"
        assert url.endswith("/a2a/assistant-1")
        assert headers["Accept"] == "text/event-stream"
        assert json.loads(body)["method"] == "message/stream"
        return 200, {"content-type": "text/event-stream"}, b'data: {"event":"task"}\n\ndata: [DONE]\n'

    client = LangSmithAPIClient(api_key="secret", transport=transport)
    assert list(client.a2a_stream("assistant-1", body={"method": "message/stream"})) == [
        {"event": "task"}
    ]


def test_langsmith_agent_server_thread_and_system_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        if "/stream?" in url:
            return 200, {"content-type": "text/event-stream"}, b'data: {"event":"run"}\n\n'
        if url.endswith("/metrics?format=prometheus"):
            return 200, {"content-type": "text/plain"}, b"agent_runs_total 1\n"
        if url.endswith("/docs"):
            return 200, {"content-type": "text/html"}, b"<html>docs</html>"
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.thread_history("thread-1", body={"limit": 1}) == {"ok": True}
    assert list(client.join_thread_stream("thread-1", stream_modes=["lifecycle"], last_event_id="7")) == [
        {"event": "run"}
    ]
    assert client.health_check(check_db=True) == {"ok": True}
    assert client.server_info() == {"ok": True}
    assert client.api_documentation() == b"<html>docs</html>"
    assert client.system_metrics() == b"agent_runs_total 1\n"

    assert calls[0][1].endswith("/threads/thread-1/history")
    assert calls[1][1].endswith("/threads/thread-1/stream?stream_modes=lifecycle")
    assert calls[1][2]["Last-Event-ID"] == "7"
    assert calls[2][1].endswith("/ok?check_db=1")
    assert calls[5][1].endswith("/metrics?format=prometheus")


def test_langsmith_dataset_and_example_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if "/datasets?" in url and "cursor=next" in url:
            payload = {"datasets": [{"id": "dataset-2"}]}
        elif "/datasets?" in url:
            payload = {"datasets": [{"id": "dataset-1"}], "next_cursor": "next"}
        else:
            payload = {"ok": True}
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.create_dataset(
        name="refunds",
        description="Refund cases",
        data_type="kv",
        metadata={"team": "support"},
    ) == {"ok": True}
    assert client.create_example(
        dataset_id="dataset-1",
        inputs={"question": "double charged"},
        outputs={"eligible": True},
        metadata={"source": "fixture"},
        example_id="example-1",
        name="Double charge",
    ) == {"ok": True}
    assert list(client.iter_datasets(query={"name": "refunds"})) == [
        {"id": "dataset-1"},
        {"id": "dataset-2"},
    ]
    assert client.get_dataset("dataset-1") == {"ok": True}
    assert client.update_dataset("dataset-1", body={"description": "Updated"}) == {"ok": True}
    assert client.delete_dataset("dataset-1") == {"ok": True}
    assert client.list_examples(query={"dataset_id": "dataset-1"}) == {"ok": True}
    assert client.get_example("example-1") == {"ok": True}
    assert client.update_example("example-1", body={"metadata": {"reviewed": True}}) == {"ok": True}
    assert client.delete_example("example-1") == {"ok": True}

    create_body = json.loads(calls[0][2])
    example_body = json.loads(calls[1][2])
    assert create_body["metadata"] == {"team": "support"}
    assert example_body["outputs"] == {"eligible": True}
    assert calls[2][1].endswith("/datasets?name=refunds")
    assert calls[3][1].endswith("/datasets?name=refunds&cursor=next")


def test_langsmith_agent_connection_helpers_preserve_agent_scoped_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.create_agent_connection(
        "agent-1", body={"provider": "github", "scopes": ["repo"]}
    ) == {"ok": True}
    assert client.list_agent_connections("agent-1") == {"ok": True}
    assert client.remove_agent_connection("agent-1", "connection-1") == {"ok": True}

    assert calls[0][0:2] == ("POST", "https://example.test/v2/auth/agents/agent-1/connections")
    assert json.loads(calls[0][2]) == {"provider": "github", "scopes": ["repo"]}
    assert calls[1][1].endswith("/v2/auth/agents/agent-1/connections")
    assert calls[2][1].endswith("/v2/auth/agents/agent-1/connections/connection-1")


def test_langsmith_platform_tool_registry_helpers_preserve_id_and_handle_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    assert client.create_tool(body={"handle": "refund_lookup", "description": "Lookup refunds"}) == {
        "ok": True
    }
    assert client.list_tools(query={"limit": 10}) == {"ok": True}
    assert client.get_tool_by_id("tool-1") == {"ok": True}
    assert client.get_tool_by_handle("refund_lookup") == {"ok": True}
    assert client.update_tool_by_id("tool-1", body={"description": "Updated"}) == {"ok": True}
    assert client.update_tool_by_handle("refund_lookup", body={"enabled": False}) == {"ok": True}
    assert client.delete_tool_by_id("tool-1") is None
    assert client.delete_tool_by_handle("refund_lookup") is None

    assert calls[0][0:2] == ("POST", "https://example.test/api/v1/platform/tools")
    assert json.loads(calls[0][2])["handle"] == "refund_lookup"
    assert calls[1][1].endswith("/api/v1/platform/tools?limit=10")
    assert calls[2][1].endswith("/api/v1/platform/tools/id/tool-1")
    assert calls[3][1].endswith("/api/v1/platform/tools/refund_lookup")


def test_langsmith_async_auth_tools_and_store_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        connection = await client.acreate_agent_connection(
            "agent-1", body={"provider": "github", "scopes": ["repo"]}
        )
        connections = await client.alist_agent_connections("agent-1")
        await client.aremove_agent_connection("agent-1", "connection-1")
        tool = await client.acreate_tool(body={"handle": "refund_lookup"})
        await client.alist_tools(query={"limit": 10})
        await client.aget_tool_by_id("tool-1")
        await client.aget_tool_by_handle("refund_lookup")
        await client.aupdate_tool_by_id("tool-1", body={"enabled": True})
        await client.aupdate_tool_by_handle("refund_lookup", body={"enabled": False})
        await client.adelete_tool_by_id("tool-1")
        await client.adelete_tool_by_handle("refund_lookup")
        stored = await client.astore_put(namespace=["support"], key="customer-1", value={"tier": "gold"})
        fetched = await client.astore_get(namespace=["support"], key="customer-1")
        searched = await client.astore_search(namespace_prefix=["support"], query={"limit": 5})
        await client.astore_delete(namespace=["support"], key="customer-1")
        return connection, connections, tool, stored, fetched, searched

    values = asyncio.run(collect())
    assert values == ({"ok": True},) * 6
    assert calls[0][1].endswith("/v2/auth/agents/agent-1/connections")
    assert calls[3][1].endswith("/api/v1/platform/tools")
    assert calls[4][1].endswith("/api/v1/platform/tools?limit=10")
    assert calls[11][1].endswith("/store/items")
    assert json.loads(calls[11][2]) == {
        "namespace": ["support"],
        "key": "customer-1",
        "value": {"tier": "gold"},
    }
    assert calls[13][1].endswith("/store/items/search")
    assert json.loads(calls[13][2]) == {"namespace_prefix": ["support"], "limit": 5}


def test_langsmith_async_fleet_deepagents_and_mcp_helpers_preserve_prefixes_and_cursors():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if "/agents?" in url:
            payload = (
                {"items": [{"id": "agent-2"}]}
                if "cursor=next" in url
                else {"items": [{"id": "agent-1"}], "next_cursor": "next"}
            )
        elif "/threads?" in url:
            payload = (
                {"items": [{"id": "thread-2"}]}
                if "cursor=next" in url
                else {"items": [{"id": "thread-1"}], "next_cursor": "next"}
            )
        else:
            status = 204 if method == "DELETE" else 200
            payload = None if status == 204 else {"ok": True}
            return status, {"content-type": "application/json"}, b"" if payload is None else json.dumps(payload).encode()
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    async def collect():
        agents = [item async for item in client.aiter_agents(query={"name": "refund"})]
        managed_agents = await client.alist_agents(path_prefix="/v1/deepagents")
        fetched = await client.aget_agent("agent-1", include_files=True)
        created = await client.acreate_agent(body={"name": "refunds"})
        await client.aupdate_agent("agent-1", body={"enabled": False})
        await client.adelete_agent("agent-1")
        threads = [item async for item in client.aiter_fleet_threads(query={"page_size": 10})]
        await client.aget_fleet_thread("thread-1")
        await client.aupdate_fleet_thread("thread-1", body={"title": "Refunds"})
        await client.adelete_fleet_thread("thread-1")
        servers = await client.alist_mcp_servers()
        templates = await client.alist_trigger_templates()
        await client.acreate_mcp_server(body={"name": "search"})
        await client.aget_mcp_server("mcp-1")
        await client.aupdate_mcp_server("mcp-1", body={"enabled": True})
        await client.adelete_mcp_server("mcp-1")
        return agents, managed_agents, fetched, created, threads, servers, templates

    values = asyncio.run(collect())
    assert [item["id"] for item in values[0]] == ["agent-1", "agent-2"]
    assert values[1:4] == ({"ok": True},) * 3
    assert [item["id"] for item in values[4]] == ["thread-1", "thread-2"]
    assert values[5:] == ({"ok": True},) * 2
    assert calls[0][1].endswith("/v1/fleet/agents?name=refund")
    assert calls[1][1].endswith("/v1/fleet/agents?name=refund&cursor=next")
    assert calls[3][1].endswith("/v1/fleet/agents/agent-1?include_files=true")
    assert calls[7][1].endswith("/v1/fleet/threads?page_size=10")
    assert calls[8][1].endswith("/v1/fleet/threads?page_size=10&cursor=next")
    assert any(url.endswith("/v1/deepagents/mcp-servers") for _, url, _ in calls)


def test_langsmith_agent_server_helpers_cover_threads_runs_assistants_and_store():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.search_threads(body={"metadata": {"team": "support"}})
    client.count_threads()
    client.delete_thread("thread-1")
    client.thread_history("thread-1")
    client.resolve_interrupt("thread-1")
    client.get_assistant("assistant-1")
    client.delete_assistant("assistant-1")
    client.search_assistants()
    client.set_latest_assistant_version("assistant-1", 2)
    client.get_run("thread-1", "run-1")
    client.cancel_run("thread-1", "run-1")
    client.delete_run("thread-1", "run-1")
    client.search_runs()
    client.store_put(namespace=["support"], key="customer-1", value={"tier": "gold"})
    client.store_get(namespace=["support"], key="customer-1")
    client.store_search(namespace_prefix=["support"])
    client.store_delete(namespace=["support"], key="customer-1")

    assert calls[0] == (
        "POST",
        "https://example.test/threads/search",
        b'{"metadata": {"team": "support"}}',
    )
    assert calls[-1][0:2] == (
        "DELETE",
        "https://example.test/store/items?namespace=support&key=customer-1",
    )


def test_langsmith_thread_pruning_and_checkpoint_state_helpers_preserve_native_paths():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.prune_threads(thread_ids=["thread-1"], strategy="keep_latest")
    client.get_thread_history("thread-1", limit=2)
    client.get_thread_state_at_checkpoint("thread-1", "checkpoint-1", subgraphs=True)
    client.get_thread_state_at_checkpoint_body(
        "thread-1",
        body={"checkpoint": {"checkpoint_id": "checkpoint-1"}, "subgraphs": False},
    )

    assert calls[0][0:2] == ("POST", "https://example.test/threads/prune")
    assert json.loads(calls[0][2]) == {"thread_ids": ["thread-1"], "strategy": "keep_latest"}
    assert calls[1][1].endswith("/threads/thread-1/history?limit=2")
    assert calls[2][1].endswith("/threads/thread-1/state/checkpoint-1?subgraphs=True")
    assert calls[3][1].endswith("/threads/thread-1/state/checkpoint")


def test_langsmith_agent_helpers_support_fleet_and_managed_deep_agents():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.list_agents(query={"page_size": 10, "audience": "tenant"})
    client.get_agent("fleet-1", include_files=True)
    client.create_agent(body={"name": "refunds"})
    client.update_agent("fleet-1", body={"description": "Refund assistant"})
    client.delete_agent("fleet-1")
    client.list_agents(path_prefix="/v1/deepagents")

    assert calls[0][1].endswith("/v1/fleet/agents?page_size=10&audience=tenant")
    assert calls[1][1].endswith("/v1/fleet/agents/fleet-1?include_files=true")
    assert calls[2][0:2] == ("POST", "https://example.test/v1/fleet/agents")
    assert calls[5][1] == "https://example.test/v1/deepagents/agents"


def test_langsmith_cursor_iterators_preserve_fleet_filters():
    calls = []

    def transport(method, url, headers, body):
        del method, headers, body
        calls.append(url)
        if "/agents" in url:
            if "cursor=agent-next" in url:
                payload = b'{"items":[{"id":"agent-2"}],"next_cursor":null}'
            else:
                payload = b'{"items":[{"id":"agent-1"}],"next_cursor":"agent-next"}'
        else:
            if "cursor=thread-next" in url:
                payload = b'{"items":[{"id":"thread-2"}],"next_cursor":null}'
            else:
                payload = b'{"items":[{"id":"thread-1"}],"next_cursor":"thread-next"}'
        return 200, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)

    assert [item["id"] for item in client.iter_agents(query={"name": "refund"})] == [
        "agent-1",
        "agent-2",
    ]
    assert [item["id"] for item in client.iter_fleet_threads(query={"page_size": 10})] == [
        "thread-1",
        "thread-2",
    ]
    client.list_trigger_templates()
    assert "name=refund" in calls[0]
    assert "cursor=agent-next" in calls[1]
    assert "page_size=10" in calls[2]
    assert "cursor=thread-next" in calls[3]
    assert calls[4].endswith("/v1/fleet/trigger-templates")


def test_langsmith_feedback_fleet_thread_and_mcp_helpers_preserve_native_shapes():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    client.create_feedback(
        run_id="run-1",
        key="user-rating",
        score=1,
        comment="Helpful",
        source_info={"origin": "app"},
    )
    client.list_feedback(query={"run_id": "run-1"})
    client.get_feedback("feedback-1")
    client.update_feedback("feedback-1", body={"comment": "Updated"})
    client.delete_feedback("feedback-1")
    client.list_fleet_threads(query={"page_size": 10})
    client.get_fleet_thread("thread-1")
    client.update_fleet_thread("thread-1", body={"title": "Refunds"})
    client.delete_fleet_thread("thread-1")
    client.list_mcp_servers()
    client.create_mcp_server(body={"name": "search", "url": "https://mcp.example.test"})
    client.get_mcp_server("mcp-1")
    client.update_mcp_server("mcp-1", body={"name": "search-v2"})
    client.delete_mcp_server("mcp-1")

    assert calls[0] == (
        "POST",
        "https://example.test/api/v1/feedback",
        b'{"run_id": "run-1", "key": "user-rating", "score": 1, "comment": "Helpful", "source_info": {"origin": "app"}}',
    )
    assert calls[1][1].endswith("/api/v1/feedback?run_id=run-1")
    assert calls[5][1].endswith("/v1/fleet/threads?page_size=10")
    assert calls[9][1] == "https://example.test/v1/deepagents/mcp-servers"


def test_langsmith_control_plane_client_covers_deployment_and_revision_lifecycle():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok": true}'

    client = LangSmithControlPlaneClient(
        api_key="secret",
        tenant_id="workspace-1",
        base_url="https://control.example.test",
        transport=transport,
    )
    client.list_deployments(name_contains="refund")
    client.create_deployment(body={"name": "refunds"})
    client.get_deployment("deployment-1")
    client.patch_deployment("deployment-1", body={"source_config": {"build_on_push": True}})
    client.list_revisions("deployment-1")
    client.get_revision("deployment-1", "revision-1")
    client.create_deployment_revision("deployment-1", body={"config": {}})
    client.redeploy_revision("deployment-1", "revision-1")
    client.interrupt_deployment_revision("deployment-1", "revision-1")
    client.list_deployment_logs("deployment-1", query={"limit": 10})
    client.list_revision_logs("deployment-1", "revision-1", query={"limit": 10})
    client.list_deployment_log_entries(
        deployment_id="deployment-1",
        revision_id="revision-1",
        log_type="BUILD",
        sort_order="asc",
    )
    client.patch_deployment_resource_tiers("deployment-1", body={"cpu": 2})
    client.patch_deployment_tier("deployment-1", body={"tier": "professional"})
    client.get_free_deployment_count()
    client.create_listener(body={"name": "refunds", "url": "https://example.test/hook"})
    client.list_listeners(query={"limit": 10})
    client.get_listener("listener-1")
    client.patch_listener("listener-1", body={"enabled": False})
    client.delete_listener("listener-1")
    assert client.delete_deployment("deployment-1") is None
    client.delete_deployments(["deployment-1", "deployment-2"])

    assert calls[0][0:2] == (
        "GET",
        "https://control.example.test/v2/deployments?name_contains=refund",
    )
    assert all(call[2]["X-Api-Key"] == "secret" for call in calls)
    assert all(call[2]["X-Tenant-Id"] == "workspace-1" for call in calls)
    assert calls[1][0:2] == ("POST", "https://control.example.test/v2/deployments")
    assert calls[6][1].endswith("/v2/deployments/deployment-1/revisions")
    assert any(call[1].endswith("/revisions/revision-1/redeploy") for call in calls)
    assert any(call[1].endswith("/revisions/revision-1/interruption") for call in calls)
    assert any("/v2/deployment-logs?" in call[1] for call in calls)
    assert any(call[1].endswith("/v2/listeners") and call[0] == "POST" for call in calls)
    assert any(call[1].endswith("/v2/listeners?limit=10") for call in calls)
    assert any(call[1].endswith("/v2/listeners/listener-1") for call in calls)
    delete_many = next(call for call in calls if call[0] == "DELETE" and "/v2/deployments?" in call[1])
    assert "deployment_ids=deployment-1" in delete_many[1]
    assert "deployment_ids=deployment-2" in delete_many[1]


def test_langsmith_control_plane_async_lifecycle_preserves_tenant_and_native_paths():
    calls = []

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        status = 204 if method == "DELETE" else 200
        payload = b"" if status == 204 else b'{"ok": true}'
        return status, {"content-type": "application/json"}, payload

    client = LangSmithControlPlaneClient(
        api_key="secret",
        tenant_id="workspace-1",
        base_url="https://control.example.test",
        transport=transport,
    )

    async def collect():
        values = [
            await client.alist_deployments(name_contains="refund"),
            await client.acreate_deployment(body={"name": "refunds"}),
            await client.aget_deployment("deployment-1"),
            await client.apatch_deployment("deployment-1", body={"enabled": True}),
            await client.alist_revisions("deployment-1"),
            await client.aget_revision("deployment-1", "revision-1"),
            await client.acreate_deployment_revision("deployment-1", body={"config": {}}),
            await client.aredeploy_revision("deployment-1", "revision-1"),
            await client.ainterrupt_deployment_revision("deployment-1", "revision-1"),
            await client.alist_deployment_logs("deployment-1", query={"limit": 10}),
            await client.alist_revision_logs("deployment-1", "revision-1", query={"limit": 10}),
            await client.alist_deployment_log_entries(
                deployment_id="deployment-1", revision_id="revision-1", log_type="BUILD"
            ),
            await client.apatch_deployment_resource_tiers("deployment-1", body={"cpu": 2}),
            await client.apatch_deployment_tier("deployment-1", body={"tier": "professional"}),
            await client.aget_free_deployment_count(),
            await client.acreate_listener(body={"name": "refunds"}),
            await client.alist_listeners(query={"limit": 10}),
            await client.aget_listener("listener-1"),
            await client.apatch_listener("listener-1", body={"enabled": False}),
            await client.adelete_listener("listener-1"),
            await client.adelete_deployment("deployment-1"),
            await client.adelete_deployments(["deployment-1", "deployment-2"]),
        ]
        return values

    values = asyncio.run(collect())
    assert all(value in ({"ok": True}, None) for value in values)
    assert all(call[2]["X-Tenant-Id"] == "workspace-1" for call in calls)
    assert calls[0][1].endswith("/v2/deployments?name_contains=refund")
    assert any(call[1].endswith("/v2/deployment-logs?deployment_id=deployment-1&revision_id=revision-1&log_type=BUILD") for call in calls)
    assert calls[-1][1].endswith("/v2/deployments?deployment_ids=deployment-1&deployment_ids=deployment-2")


def test_langsmith_control_plane_requires_workspace_identity():
    try:
        LangSmithControlPlaneClient(api_key="secret", base_url="https://example.test")
    except ValueError as exc:
        assert "LANGSMITH_TENANT_ID" in str(exc)
    else:
        raise AssertionError("control-plane client should require a tenant id")
