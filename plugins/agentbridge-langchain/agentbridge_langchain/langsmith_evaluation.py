"""Optional LangSmith dataset and evaluation integration.

This module is kept in the LangChain plugin so the core SDK remains free of a
LangSmith dependency. It accepts a client-like object for deterministic tests
and uses the official ``langsmith.Client`` only when no client is supplied.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from agentbridge.evaluation import EvaluationExample
from agentbridge.runner import run_agent
from agentbridge.types import AgentSpec


def _client_or_default(client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LangSmith integration requires langsmith. "
            "Install `agentbridge-langchain[observability]`."
        ) from exc
    return Client()


def publish_dataset(
    examples: Iterable[EvaluationExample],
    *,
    dataset_name: str,
    description: str | None = None,
    client: Any | None = None,
) -> Any:
    """Create a LangSmith dataset and upload AgentBridge examples."""

    native_client = _client_or_default(client)
    dataset = native_client.create_dataset(
        dataset_name,
        description=description,
        metadata={"source": "agentbridge", "contract": "EvaluationExample"},
    )
    native_client.create_examples(
        dataset_id=getattr(dataset, "id", None),
        examples=[
            {
                "inputs": {"input": example.input, "context": example.context},
                "outputs": {"expected_output": example.expected_output},
                "metadata": {"source": "agentbridge", **example.metadata},
            }
            for example in examples
        ],
    )
    return dataset


def evaluate_on_langsmith(
    agent: AgentSpec,
    *,
    backend: str,
    dataset: Iterable[EvaluationExample],
    dataset_name: str,
    evaluators: Sequence[Any] | None = None,
    client: Any | None = None,
    experiment_prefix: str | None = None,
    **evaluate_kwargs: Any,
) -> Any:
    """Publish a dataset and execute it through LangSmith's evaluator runner."""

    native_client = _client_or_default(client)
    publish_dataset(
        dataset,
        dataset_name=dataset_name,
        client=native_client,
    )

    def target(inputs: dict[str, Any]) -> Any:
        return run_agent(
            agent,
            backend=backend,
            input=str(inputs.get("input", "")),
            context=dict(inputs.get("context") or {}),
        ).output

    return native_client.evaluate(
        target,
        data=dataset_name,
        evaluators=list(evaluators or []),
        experiment_prefix=experiment_prefix,
        **evaluate_kwargs,
    )

