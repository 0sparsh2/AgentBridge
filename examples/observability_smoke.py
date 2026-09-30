"""Explicitly gated LangSmith and Langfuse smoke checks."""

from __future__ import annotations

import os
from typing import Any


def run_observability_smoke_matrix_from_env() -> dict[str, dict[str, Any]]:
    """Run credentialed observability checks only after explicit opt-in."""

    if os.getenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE") != "1":
        return {
            name: {
                "ok": False,
                "status": "skipped",
                "reason": "Set AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 to run live smoke tests.",
            }
            for name in ("langsmith_api", "langsmith_prompt", "langfuse_api", "langfuse_runtime")
        }
    return {
        "langsmith_api": run_langsmith_api_smoke_from_env(),
        "langsmith_prompt": run_langsmith_prompt_smoke_from_env(),
        "langfuse_api": run_langfuse_api_smoke_from_env(),
        "langfuse_runtime": run_langfuse_runtime_smoke_from_env(),
    }


def run_langsmith_api_smoke_from_env() -> dict[str, Any]:
    """Check authenticated LangSmith API access without creating a run."""

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return {"ok": False, "status": "skipped", "missing_env": ["LANGSMITH_API_KEY"]}
    try:
        from agentbridge_langchain.langsmith_api import LangSmithAPIClient

        info = LangSmithAPIClient(
            api_key=api_key,
            base_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
        ).request_json("GET", "/info")
        return {"ok": True, "status": "verified", "response_keys": _response_keys(info)}
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }


def run_langfuse_runtime_smoke_from_env() -> dict[str, Any]:
    """Run one offline LangChain invocation with the Langfuse callback enabled."""

    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return {"ok": False, "status": "skipped", "missing_env": missing}
    try:
        from agentbridge import AgentSpec, RunInput
        from agentbridge.extensions.langchain import LangChainExtension
        from agentbridge_langchain.adapter import Adapter

        agent = LangChainExtension.with_config(
            AgentSpec(
                name="agentbridge_langfuse_smoke",
                instructions="Reply to the user.",
                model="agentbridge/offline",
            ),
            observability={
                "langfuse": {
                    "enabled": True,
                    "user_id": os.getenv("LANGFUSE_USER_ID", "agentbridge-smoke"),
                }
            },
        )
        result = Adapter().run(Adapter().compile(agent), RunInput(input="hello"))
        return {
            "ok": bool(result.output),
            "status": "verified" if result.output else "failed",
            "backend": result.backend,
            "output_present": bool(result.output),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }


def run_langfuse_api_smoke_from_env() -> dict[str, Any]:
    """Check authenticated Langfuse API access without creating an observation."""

    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return {"ok": False, "status": "skipped", "missing_env": missing}
    try:
        from agentbridge_langchain.langfuse_api import LangfuseAPIClient

        response = LangfuseAPIClient(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        ).request_json("GET", "/api/public/health")
        return {"ok": True, "status": "verified", "response_keys": _response_keys(response)}
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }


def run_langsmith_prompt_smoke_from_env() -> dict[str, Any]:
    """Pull one configured LangSmith prompt without exposing its content."""

    missing = [
        name
        for name, value in {
            "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
            "AGENTBRIDGE_LANGSMITH_PROMPT": os.getenv("AGENTBRIDGE_LANGSMITH_PROMPT"),
        }.items()
        if not value
    ]
    if missing:
        return {"ok": False, "status": "skipped", "missing_env": missing}
    try:
        from agentbridge_langchain.langsmith_prompts import pull_prompt

        prompt = pull_prompt(os.environ["AGENTBRIDGE_LANGSMITH_PROMPT"])
        return {"ok": prompt is not None, "status": "verified", "prompt_type": type(prompt).__name__}
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc)[:500],
        }


def _response_keys(value: Any) -> list[str]:
    return sorted(value) if isinstance(value, dict) else []


if __name__ == "__main__":
    import json

    print(json.dumps(run_observability_smoke_matrix_from_env(), indent=2, sort_keys=True))
