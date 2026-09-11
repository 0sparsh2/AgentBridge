"""Backend comparison helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentbridge.registry import inspect_backend, list_adapters

STATUS_SCORES = {
    "full": 4,
    "partial": 2,
    "extension": 2,
    "native_only": 1,
    "unsupported": 0,
}


class BackendComparison(BaseModel):
    """Capability fit for a backend."""

    backend: str
    score: int
    max_score: int
    features: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    notes: dict[str, str] = Field(default_factory=dict)

    @property
    def fit(self) -> str:
        if self.max_score == 0:
            return "unknown"
        ratio = self.score / self.max_score
        if ratio >= 0.9:
            return "strong"
        if ratio >= 0.6:
            return "medium"
        return "weak"

    def as_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["fit"] = self.fit
        return data


def compare_backends(
    required_features: list[str],
    *,
    backends: list[str] | None = None,
) -> list[BackendComparison]:
    """Compare backends against required capabilities."""

    selected_backends = backends or list_adapters()
    comparisons: list[BackendComparison] = []
    max_score = len(required_features) * STATUS_SCORES["full"]

    for backend in selected_backends:
        capabilities = inspect_backend(backend)
        feature_statuses: dict[str, str] = {}
        missing: list[str] = []
        score = 0
        notes: dict[str, str] = {}

        for feature in required_features:
            status = capabilities.status(feature)
            feature_statuses[feature] = status
            score += STATUS_SCORES[status]
            if status == "unsupported":
                missing.append(feature)
            if feature in capabilities.notes:
                notes[feature] = capabilities.notes[feature]

        comparisons.append(
            BackendComparison(
                backend=backend,
                score=score,
                max_score=max_score,
                features=feature_statuses,
                missing=missing,
                notes=notes,
            )
        )

    return sorted(comparisons, key=lambda comparison: (-comparison.score, comparison.backend))
