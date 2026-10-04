"""Optional provider integrations for LangChain-compatible runtimes."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def observability_metadata(
    config: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build provider-neutral trace metadata with LangSmith/Langfuse correlation."""

    result = dict(config.get("metadata", {}))
    result.update(metadata or {})
    observability = config.get("observability", {})
    langfuse = observability.get("langfuse", {})
    langsmith = observability.get("langsmith", {})
    if session_id:
        result["session_id"] = session_id
    if langfuse.get("enabled"):
        if langfuse.get("session_id") or session_id:
            result["langfuse_session_id"] = langfuse.get("session_id") or session_id
        if langfuse.get("user_id"):
            result["langfuse_user_id"] = langfuse["user_id"]
        if langfuse.get("trace_id"):
            result["langfuse_trace_id"] = langfuse["trace_id"]
    if langsmith.get("enabled"):
        if langsmith.get("project_name"):
            result["langsmith_project"] = langsmith["project_name"]
        if langsmith.get("trace_id"):
            result["langsmith_trace_id"] = langsmith["trace_id"]
    return result


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
        callback_options = {
            key: langfuse[key]
            for key in (
                "public_key",
                "secret_key",
                "host",
                "release",
                "version",
                "environment",
                "session_id",
                "user_id",
                "trace_id",
                "debug",
            )
            if langfuse.get(key) is not None
        }
        try:
            callbacks.append(CallbackHandler(**callback_options))
        except TypeError:
            # Older Langfuse SDK releases accepted only environment-driven options.
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
