"""Publish AgentBridge evaluation datasets and scores to Langfuse."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentbridge.evaluation import EvaluationExample, EvaluationReport

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
        )
    return dataset


def publish_report_scores(
    report: EvaluationReport,
    *,
    trace_ids: Mapping[int, str],
    client: LangfuseAPIClient,
) -> list[Any]:
    """Publish normalized evaluation scores against known Langfuse trace IDs."""

    published: list[Any] = []
    for case in report.cases:
        trace_id = trace_ids.get(case.index)
        if not trace_id:
            continue
        for score in case.scores:
            if score.score is None:
                continue
            published.append(
                client.create_score(
                    trace_id=trace_id,
                    name=score.key,
                    value=score.score,
                    comment=score.comment,
                )
            )
    return published


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
        )
    return dataset


async def apublish_report_scores(
    report: EvaluationReport,
    *,
    trace_ids: Mapping[int, str],
    client: LangfuseAPIClient,
) -> list[Any]:
    """Async publish normalized evaluation scores against Langfuse traces."""

    published: list[Any] = []
    for case in report.cases:
        trace_id = trace_ids.get(case.index)
        if not trace_id:
            continue
        for score in case.scores:
            if score.score is None:
                continue
            published.append(
                await client.acreate_score(
                    trace_id=trace_id,
                    name=score.key,
                    value=score.score,
                    comment=score.comment,
                )
            )
    return published
