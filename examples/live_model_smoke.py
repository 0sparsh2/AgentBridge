"""Run an explicitly gated OpenAI-compatible model smoke test."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def run_openai_compatible_smoke(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str = "Reply with exactly: AgentBridge smoke test ok",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint."""

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "model": model,
            "base_url": base_url,
            "error": "HTTPError",
            "message": _safe_error_body(body),
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "status": None,
            "model": model,
            "base_url": base_url,
            "error": "URLError",
            "message": str(exc.reason),
        }

    data = json.loads(body)
    content = _first_message_content(data)
    return {
        "ok": bool(content),
        "status": status,
        "model": model,
        "base_url": base_url,
        "content": content,
        "usage": data.get("usage", {}),
        "raw_keys": sorted(data),
    }


def run_nvidia_nim_smoke_from_env() -> dict[str, Any]:
    """Run a gated NVIDIA NIM smoke test from environment variables."""

    if os.getenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE") != "1":
        return {
            "ok": False,
            "status": "skipped",
            "reason": "Set AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 to run live smoke tests.",
        }
    api_key = os.getenv("NVIDIA_NIM_API_KEY")
    base_url = os.getenv("NVIDIA_NIM_API_BASE") or os.getenv("NVIDIA_NIM_BASE_URL")
    model = (
        os.getenv("NVIDIA_MODEL")
        or os.getenv("NVIDIA_NIM_MODEL")
        or "deepseek-ai/deepseek-v4-flash-0731"
    )
    missing = [
        name
        for name, value in {
            "NVIDIA_NIM_API_KEY": api_key,
            "NVIDIA_NIM_API_BASE or NVIDIA_NIM_BASE_URL": base_url,
            "NVIDIA_MODEL or NVIDIA_NIM_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        return {
            "ok": False,
            "status": "skipped",
            "missing_env": missing,
        }
    return run_openai_compatible_smoke(
        api_key=str(api_key),
        base_url=str(base_url),
        model=str(model),
    )


def run_provider_smoke_matrix_from_env() -> dict[str, dict[str, Any]]:
    """Run gated OpenAI-compatible smoke tests for NIM, OpenRouter, and Ollama."""

    route_names = ("nvidia_nim", "openrouter", "ollama")
    if os.getenv("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE") != "1":
        return {
            name: {
                "ok": False,
                "status": "skipped",
                "reason": "Set AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 to run live smoke tests.",
            }
            for name in route_names
        }

    routes = {
        "nvidia_nim": {
            "api_key": os.getenv("NVIDIA_NIM_API_KEY"),
            "base_url": os.getenv("NVIDIA_NIM_API_BASE") or os.getenv("NVIDIA_NIM_BASE_URL"),
            "model": os.getenv("NVIDIA_MODEL") or os.getenv("NVIDIA_NIM_MODEL"),
        },
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        },
        "ollama": {
            "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        },
    }
    results: dict[str, dict[str, Any]] = {}
    for name, route in routes.items():
        missing: list[str] = []
        if not route["api_key"]:
            missing.append(
                "NVIDIA_NIM_API_KEY" if name == "nvidia_nim" else f"{name.upper()}_API_KEY"
            )
        if name == "nvidia_nim" and not route["base_url"]:
            missing.append("NVIDIA_NIM_API_BASE or NVIDIA_NIM_BASE_URL")
        if missing:
            results[name] = {"ok": False, "status": "skipped", "missing_env": missing}
            continue
        results[name] = run_openai_compatible_smoke(
            api_key=str(route["api_key"]),
            base_url=str(route["base_url"]),
            model=str(route["model"]),
        )
    return results


def _first_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _safe_error_body(body: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:500]
    if isinstance(data, dict):
        error = data.get("error", data)
        return json.dumps(error, sort_keys=True)[:500]
    return body[:500]


if __name__ == "__main__":
    print(json.dumps(run_nvidia_nim_smoke_from_env(), indent=2, sort_keys=True))
