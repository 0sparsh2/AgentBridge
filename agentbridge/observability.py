"""Optional provider integrations for LangChain-compatible runtimes."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def callbacks_for_config(config: dict[str, Any]) -> list[Any]:
    """Return user callbacks plus enabled optional provider callbacks."""

    callbacks = list(config.get("callbacks") or [])
    langfuse = config.get("observability", {}).get("langfuse", {})
    if langfuse.get("enabled"):
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:  # pragma: no cover - optional integration
            raise ImportError(
                "Langfuse observability is enabled but langfuse is not installed. "
                "Install the relevant adapter with its observability extra."
            ) from exc
        callbacks.append(CallbackHandler())
    return callbacks


def langsmith_context(config: dict[str, Any]) -> Any:
    """Return a LangSmith tracing context when explicitly enabled."""

    langsmith = config.get("observability", {}).get("langsmith", {})
    if not langsmith.get("enabled"):
        return nullcontext()
    try:
        from langsmith.run_helpers import tracing_context
    except ImportError as exc:  # pragma: no cover - optional integration
        raise ImportError(
            "LangSmith observability is enabled but langsmith is not installed."
        ) from exc
    return tracing_context(
        project_name=langsmith.get("project_name"),
        tags=langsmith.get("tags") or config.get("observability", {}).get("tags"),
        metadata=langsmith.get("metadata") or config.get("metadata", {}),
        enabled=True,
    )
