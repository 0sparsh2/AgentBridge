"""Publish AgentBridge evaluation datasets and scores to Langfuse."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentbridge.evaluation import EvaluationExample, EvaluationReport, EvaluationScore

from .langfuse_api import LangfuseAPIClient


def publish_dataset(
    examples: Iterable[EvaluationExample],
    *,
    dataset_name: str,
    description: str | None = None,
    client: LangfuseAPIClient,
) -> Any:
    """Create a Langfuse dataset and upload AgentBridge examples."""

    materialized = list(examples)
    dataset = client.create_dataset(name=dataset_name, description=description)
    for example in materialized:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"input": example.input, "context": example.context},
            expected_output=example.expected_output,
            metadata=example.metadata or None,
        )
    return dataset


def publish_report_scores(
    report: EvaluationReport,
    *,
    trace_ids: Mapping[int, str],
    client: LangfuseAPIClient,
    score_types: Mapping[str, str] | None = None,
    score_config_ids: Mapping[str, str] | None = None,
) -> list[Any]:
    """Publish normalized evaluation scores against known Langfuse trace IDs."""

    payloads = _report_score_payloads(
        report, trace_ids=trace_ids, score_types=score_types, score_config_ids=score_config_ids,
    )
    return [client.create_score(**payload) for payload in payloads]


async def apublish_dataset(
    examples: Iterable[EvaluationExample],
    *,
    dataset_name: str,
    description: str | None = None,
    client: LangfuseAPIClient,
) -> Any:
    """Async create a Langfuse dataset and upload AgentBridge examples."""

    materialized = list(examples)
    dataset = await client.acreate_dataset(name=dataset_name, description=description)
    for example in materialized:
        await client.acreate_dataset_item(
            dataset_name=dataset_name,
            input={"input": example.input, "context": example.context},
            expected_output=example.expected_output,
            metadata=example.metadata or None,
        )
    return dataset


async def apublish_report_scores(
    report: EvaluationReport,
    *,
    trace_ids: Mapping[int, str],
    client: LangfuseAPIClient,
    score_types: Mapping[str, str] | None = None,
    score_config_ids: Mapping[str, str] | None = None,
) -> list[Any]:
    """Async publish normalized evaluation scores against Langfuse traces."""

    payloads = _report_score_payloads(
        report, trace_ids=trace_ids, score_types=score_types, score_config_ids=score_config_ids,
    )
    return [await client.acreate_score(**payload) for payload in payloads]


def _report_score_payloads(
    report: EvaluationReport, *, trace_ids: Mapping[int, str], score_types: Mapping[str, str] | None,
    score_config_ids: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Validate publishable scores before starting network writes."""

    payloads = []
    for case in report.cases:
        trace_id = trace_ids.get(case.index)
        if not trace_id:
            continue
        for score in case.scores:
            payload = _score_payload(score, trace_id=trace_id, score_types=score_types)
            if payload is None:
                continue
            config_id = (score_config_ids or {}).get(score.key)
            if config_id is not None:
                payload["config_id"] = config_id
            payloads.append(payload)
    return payloads


def _score_payload(
    score: EvaluationScore, *, trace_id: str, score_types: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    """Preserve typed evaluator values and infer native types for unambiguous scalars."""

    value = score.value if score.value is not None else score.score
    if value is None:
        return None
    if not isinstance(value, (bool, str, int, float)):
        raise ValueError(f"Langfuse score {score.key!r} must contain a scalar value.")
    data_type = (score_types or {}).get(score.key)
    if data_type is None:
        if isinstance(value, bool):
            data_type = "BOOLEAN"
        elif isinstance(value, str):
            data_type = "TEXT"
        elif isinstance(value, (int, float)):
            data_type = "NUMERIC"
    payload = {
        "trace_id": trace_id, "name": score.key, "value": value,
        "comment": score.comment, "data_type": data_type,
    }
    if score.metadata:
        payload["metadata"] = dict(score.metadata)
    return payload
