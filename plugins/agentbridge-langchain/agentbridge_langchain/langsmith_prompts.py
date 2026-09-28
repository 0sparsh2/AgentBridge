"""Optional LangSmith prompt versioning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _client_or_default(client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "LangSmith prompt integration requires langsmith. "
            "Install `agentbridge-langchain[observability]`."
        ) from exc
    return Client()


def pull_prompt(
    prompt_identifier: str,
    *,
    client: Any | None = None,
    include_model: bool = True,
    skip_cache: bool = False,
) -> Any:
    """Pull a LangSmith prompt by name, tag, or commit and return it natively."""

    return _client_or_default(client).pull_prompt(
        prompt_identifier,
        include_model=include_model,
        skip_cache=skip_cache,
    )


def push_prompt(
    prompt_identifier: str,
    *,
    prompt: Any,
    client: Any | None = None,
    parent_commit_hash: str = "latest",
    tags: Sequence[str] | None = None,
    description: str | None = None,
    commit_description: str | None = None,
) -> str:
    """Push a native LangChain prompt as a new LangSmith prompt commit."""

    return _client_or_default(client).push_prompt(
        prompt_identifier,
        object=prompt,
        parent_commit_hash=parent_commit_hash,
        tags=list(tags or []),
        description=description,
        commit_description=commit_description,
    )

