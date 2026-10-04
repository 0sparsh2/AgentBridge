from agentbridge import EvaluationExample, EvaluationReport, EvaluationCase, EvaluationScore
from agentbridge_langchain.langfuse_evaluation import publish_dataset, publish_report_scores


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


def test_langfuse_evaluation_bridge_publishes_examples_and_scores():
    client = FakeClient()
    publish_dataset(
        [EvaluationExample(input="A123", expected_output={"eligible": True})],
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
    assert client.scores[0]["trace_id"] == "trace-1"
    assert published == [{"id": "score-1"}]
