"""Framework-neutral offline evaluation primitives.

The evaluation contract deliberately stays local and dependency-free. Hosted
systems such as LangSmith can be connected by using AgentBridge observability
configuration or by supplying a custom evaluator that reports to the native
client. This keeps the core usable in CI and on air-gapped test data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentbridge.runner import run_agent
from agentbridge.types import AgentSpec, RunResult


class EvaluationExample(BaseModel):
    """One input/reference pair in an evaluation dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: str = Field(min_length=1)
    expected_output: Any | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationScore(BaseModel):
    """One evaluator result for one dataset example."""

    key: str = Field(min_length=1)
    score: float | None = None
    value: Any | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationCase(BaseModel):
    """Normalized output for one executed evaluation example."""

    index: int = Field(ge=0)
    example: EvaluationExample
    result: Any
    backend: str
    scores: list[EvaluationScore] = Field(default_factory=list)
    error: str | None = None


class EvaluationReport(BaseModel):
    """Dataset-level evaluation report."""

    dataset_name: str = Field(min_length=1)
    backend: str
    cases: list[EvaluationCase] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


Evaluator = Callable[[EvaluationExample, RunResult], Any]


def evaluate_agent(
    agent: AgentSpec,
    *,
    backend: str,
    dataset: Iterable[EvaluationExample],
    evaluators: Mapping[str, Evaluator] | None = None,
    dataset_name: str = "agentbridge-evaluation",
) -> EvaluationReport:
    """Run an agent against examples and apply deterministic evaluators.

    Evaluators receive the original :class:`EvaluationExample` and normalized
    :class:`RunResult`. They may return a number, a mapping, an
    :class:`EvaluationScore`, or a list of scores.
    """

    cases: list[EvaluationCase] = []
    evaluator_map = dict(evaluators or {})
    for index, example in enumerate(dataset):
        try:
            result = run_agent(
                agent,
                backend=backend,
                input=example.input,
                context=example.context,
                metadata=example.metadata,
            )
            scores: list[EvaluationScore] = []
            for key, evaluator in evaluator_map.items():
                scores.extend(_normalize_scores(key, evaluator(example, result)))
            cases.append(
                EvaluationCase(
                    index=index,
                    example=example,
                    result=result.output,
                    backend=result.backend,
                    scores=scores,
                )
            )
        except Exception as exc:  # noqa: BLE001 - reports preserve per-case failures.
            cases.append(
                EvaluationCase(
                    index=index,
                    example=example,
                    result=None,
                    backend=backend,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return EvaluationReport(
        dataset_name=dataset_name,
        backend=backend,
        cases=cases,
        summary=_summarize(cases),
    )


def _normalize_scores(key: str, value: Any) -> list[EvaluationScore]:
    if isinstance(value, EvaluationScore):
        return [value]
    if isinstance(value, list | tuple):
        scores: list[EvaluationScore] = []
        for item in value:
            scores.extend(_normalize_scores(key, item))
        return scores
    if isinstance(value, Mapping):
        return [
            EvaluationScore(
                key=str(value.get("key", key)),
                score=_coerce_score(value.get("score")),
                value=value.get("value"),
                comment=value.get("comment"),
                metadata=dict(value.get("metadata") or {}),
            )
        ]
    if isinstance(value, bool | int | float):
        numeric = float(value)
        return [EvaluationScore(key=key, score=numeric, value=value)]
    return [EvaluationScore(key=key, value=value)]


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, bool | int | float):
        return float(value)
    return None


def _summarize(cases: list[EvaluationCase]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    for case in cases:
        for score in case.scores:
            if score.score is not None:
                scores.setdefault(score.key, []).append(score.score)
    return {
        "example_count": len(cases),
        "success_count": sum(case.error is None for case in cases),
        "error_count": sum(case.error is not None for case in cases),
        "score_means": {
            key: sum(values) / len(values) for key, values in sorted(scores.items()) if values
        },
    }
