from __future__ import annotations

from types import SimpleNamespace

from agentbridge import AgentSpec, EvaluationExample
from agentbridge_langchain.langsmith_api import LangSmithAPIClient
from agentbridge_langchain.langsmith_evaluation import evaluate_on_langsmith, publish_dataset


class FakeClient:
    def __init__(self):
        self.dataset_calls = []
        self.example_calls = []
        self.evaluate_calls = []

    def create_dataset(self, name, **kwargs):
        self.dataset_calls.append((name, kwargs))
        return SimpleNamespace(id="dataset-1")

    def create_examples(self, **kwargs):
        self.example_calls.append(kwargs)

    def evaluate(self, target, **kwargs):
        self.evaluate_calls.append((target, kwargs))
        return {"experiment": kwargs["experiment_prefix"], "output": target({"input": "A123"})}


def test_publish_dataset_maps_agentbridge_examples():
    client = FakeClient()

    dataset = publish_dataset(
        [EvaluationExample(input="Check A123", expected_output="eligible")],
        dataset_name="refunds",
        client=client,
    )

    assert dataset.id == "dataset-1"
    assert client.dataset_calls[0][0] == "refunds"
    assert client.example_calls[0]["dataset_id"] == "dataset-1"
    assert client.example_calls[0]["examples"][0]["inputs"]["input"] == "Check A123"


def test_evaluate_on_langsmith_uses_agentbridge_target():
    client = FakeClient()
    agent = AgentSpec(
        name="refunds",
        instructions="Decide refund eligibility.",
        model="agentbridge/offline",
    )

    result = evaluate_on_langsmith(
        agent,
        backend="mock",
        dataset=[EvaluationExample(input="A123")],
        dataset_name="refunds",
        client=client,
        experiment_prefix="agentbridge-refunds",
    )

    assert result["experiment"] == "agentbridge-refunds"
    assert result["output"]["input"] == "A123"
    assert client.evaluate_calls[0][1]["data"] == "refunds"


def test_publish_dataset_supports_dependency_free_langsmith_api_client():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append((method, url, body))
        if method == "POST" and url.endswith("/datasets"):
            payload = b'{"id": "dataset-1"}'
        else:
            payload = b'{"id": "example-1"}'
        return 200, {"content-type": "application/json"}, payload

    client = LangSmithAPIClient(api_key="secret", base_url="https://example.test", transport=transport)
    dataset = publish_dataset(
        [EvaluationExample(input="Check A123", expected_output="eligible", metadata={"team": "support"})],
        dataset_name="refunds",
        description="Refund regression cases",
        client=client,
    )

    assert dataset == {"id": "dataset-1"}
    assert calls[0][0:2] == ("POST", "https://example.test/api/v1/datasets")
    assert calls[1][0:2] == ("POST", "https://example.test/api/v1/examples")
