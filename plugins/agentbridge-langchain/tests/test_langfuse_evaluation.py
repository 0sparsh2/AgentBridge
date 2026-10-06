from agentbridge import EvaluationExample, EvaluationReport, EvaluationCase, EvaluationScore
import asyncio
import json

import pytest

from agentbridge_langchain import LangfuseAPIClient

from agentbridge_langchain import (
    apublish_langfuse_dataset,
    apublish_report_scores as exported_apublish_report_scores,
)
from agentbridge_langchain.langfuse_evaluation import (
    publish_dataset,
    publish_report_scores,
)


class FakeClient:
    def __init__(self):
        self.datasets = []
        self.items = []
        self.scores = []

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)
        return {"name": kwargs["name"]}

    def create_dataset_item(self, **kwargs):
        self.items.append(kwargs)
        return {"id": "item-1"}

    def create_score(self, **kwargs):
        self.scores.append(kwargs)
        return {"id": "score-1"}

    async def acreate_dataset(self, **kwargs):
        return self.create_dataset(**kwargs)

    async def acreate_dataset_item(self, **kwargs):
        return self.create_dataset_item(**kwargs)

    async def acreate_score(self, **kwargs):
        return self.create_score(**kwargs)


def test_langfuse_evaluation_bridge_publishes_examples_and_scores():
    client = FakeClient()
    publish_dataset(
        [EvaluationExample(input="A123", expected_output={"eligible": True}, metadata={"policy": "v2"})],
        dataset_name="refunds",
        client=client,
    )
    report = EvaluationReport(
        dataset_name="refunds",
        backend="mock",
        cases=[
            EvaluationCase(
                index=0,
                example=EvaluationExample(input="A123"),
                result="yes",
                backend="mock",
                scores=[EvaluationScore(key="quality", score=0.9, comment="good")],
            )
        ],
    )
    published = publish_report_scores(report, trace_ids={0: "trace-1"}, client=client)
    assert client.items[0]["dataset_name"] == "refunds"
    assert client.items[0]["metadata"] == {"policy": "v2"}
    assert client.scores[0]["trace_id"] == "trace-1"
    assert published == [{"id": "score-1"}]


def test_async_langfuse_evaluation_bridge_publishes_examples_and_scores():
    client = FakeClient()
    report = EvaluationReport(
        dataset_name="refunds",
        backend="mock",
        cases=[
            EvaluationCase(
                index=0,
                example=EvaluationExample(input="A123"),
                result="yes",
                backend="mock",
                scores=[EvaluationScore(key="quality", score=0.9, comment="good")],
            )
        ],
    )

    async def publish():
        dataset = await apublish_langfuse_dataset(
            [EvaluationExample(input="A123", expected_output={"eligible": True}, metadata={"policy": "v2"})],
            dataset_name="refunds",
            client=client,
        )
        scores = await exported_apublish_report_scores(
            report, trace_ids={0: "trace-1"}, client=client
        )
        return dataset, scores

    dataset, scores = asyncio.run(publish())
    assert dataset == {"name": "refunds"}
    assert scores == [{"id": "score-1"}]
    assert client.items[-1]["dataset_name"] == "refunds"
    assert client.items[-1]["metadata"] == {"policy": "v2"}
    assert client.scores[-1]["trace_id"] == "trace-1"


def test_score_publication_preserves_typed_values_metadata_and_explicit_categories():
    requests = []

    def transport(method, url, headers, body):
        assert method == "POST"
        assert url.endswith("/api/public/scores")
        requests.append(json.loads(body))
        return 200, {"content-type": "application/json"}, b'{"id":"published"}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    report = EvaluationReport(dataset_name="refunds", backend="mock", cases=[
        EvaluationCase(index=0, example=EvaluationExample(input="A123"), result="yes", backend="mock",
                       scores=[
                           EvaluationScore(key="quality", score=0),
                           EvaluationScore(key="allowed", score=1, value=False),
                           EvaluationScore(key="verdict", value="reject", metadata={"policy": "v2"}),
                           EvaluationScore(key="explanation", value=""),
                           EvaluationScore(key="empty"),
                       ]),
        EvaluationCase(index=1, example=EvaluationExample(input="A456"), result="no", backend="mock",
                       scores=[EvaluationScore(key="unmapped_trace", value=True)]),
    ])
    publish_report_scores(report, trace_ids={0: "trace-1"}, client=client,
                          score_types={"verdict": "CATEGORICAL"},
                          score_config_ids={"verdict": "verdict-config"})
    synchronous = list(requests)
    asyncio.run(exported_apublish_report_scores(
        report, trace_ids={0: "trace-1"}, client=client, score_types={"verdict": "CATEGORICAL"},
        score_config_ids={"verdict": "verdict-config"},
    ))
    assert requests[4:] == synchronous
    assert [(item["name"], item["dataType"], item["value"]) for item in synchronous] == [
        ("quality", "NUMERIC", 0), ("allowed", "BOOLEAN", False),
        ("verdict", "CATEGORICAL", "reject"), ("explanation", "TEXT", ""),
    ]
    assert synchronous[2]["metadata"] == {"policy": "v2"}
    assert synchronous[2]["configId"] == "verdict-config"
    assert all("configId" not in item for index, item in enumerate(synchronous) if index != 2)
    assert all(item["traceId"] == "trace-1" for item in synchronous)


def test_non_scalar_evaluation_values_raise_instead_of_being_silently_skipped():
    report = EvaluationReport(dataset_name="refunds", backend="mock", cases=[
        EvaluationCase(index=0, example=EvaluationExample(input="A123"), result=None, backend="mock",
                       scores=[EvaluationScore(key="valid", score=0.8),
                               EvaluationScore(key="invalid", value={"reason": "nested"})]),
    ])
    client = FakeClient()
    with pytest.raises(ValueError, match="scalar value"):
        publish_report_scores(report, trace_ids={0: "trace-1"}, client=client)
    with pytest.raises(ValueError, match="scalar value"):
        asyncio.run(exported_apublish_report_scores(report, trace_ids={0: "trace-1"}, client=client))
    assert client.scores == []


def test_typed_evaluation_example_round_trips_sync_async_scores_and_subjects():
    from examples.langfuse_typed_evaluation import run_demo

    report = run_demo()
    assert report["parity"] is True
    assert [item["dataType"] for item in report["sync"]] == [
        "NUMERIC", "BOOLEAN", "CATEGORICAL", "TEXT"
    ]
    assert report["sync"][1]["value"] is False
    assert report["sync"][2]["metadata"] == {"policy": "v2"}
    assert report["sync"][0]["subject"] == {"kind": "trace", "id": "trace-demo"}
