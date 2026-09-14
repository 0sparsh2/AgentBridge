"""Inspect optional credentialed smoke-test readiness without contacting providers."""

from __future__ import annotations

import json
import os

from examples.model_routes import MODEL_ROUTES


SMOKE_ROUTES = {
    "openai_api": {
        "backend": "pydantic_ai",
        "required_env": ["OPENAI_API_KEY"],
        "notes": "Hosted OpenAI-compatible API smoke path.",
    },
    "anthropic_api": {
        "backend": "pydantic_ai",
        "required_env": ["ANTHROPIC_API_KEY"],
        "notes": "Hosted Anthropic API smoke path.",
    },
    "google_api": {
        "backend": "google_adk",
        "required_env": ["GOOGLE_API_KEY"],
        "notes": "Google ADK/Gemini credentialed smoke path.",
    },
    "local_ollama": {
        "backend": "langchain",
        "required_env": ["OLLAMA_BASE_URL"],
        "notes": "Local Ollama-compatible server smoke path.",
    },
    "openrouter": {
        "backend": "pydantic_ai",
        "required_env": ["OPENROUTER_API_KEY"],
        "notes": "OpenRouter route through provider-compatible settings.",
    },
    "nvidia_nim_openai_compatible": {
        "backend": "pydantic_ai",
        "required_env": ["NVIDIA_NIM_API_KEY"],
        "any_of_env": [["NVIDIA_NIM_API_BASE", "NVIDIA_NIM_BASE_URL"]],
        "optional_env": ["NVIDIA_MODEL", "NVIDIA_NIM_MODEL"],
        "notes": "NVIDIA NIM via OpenAI-compatible base URL.",
    },
    "custom_openai_compatible_gateway": {
        "backend": "pydantic_ai",
        "required_env": [
            "AGENTBRIDGE_GATEWAY_API_KEY",
            "AGENTBRIDGE_GATEWAY_BASE_URL",
        ],
        "notes": "Internal or third-party OpenAI-compatible gateway.",
    },
}

NATIVE_RUNTIME_SMOKES = {
    "openai_agents_native_resume": {
        "backend": "openai_agents",
        "required_env": ["OPENAI_API_KEY"],
        "notes": "Live OpenAI Agents approval interruption and backend-specific resume execution.",
    },
    "strands_agentcore_deployment": {
        "backend": "strands",
        "required_env": [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AGENTBRIDGE_STRANDS_AGENTCORE_ROLE_ARN",
        ],
        "notes": "Live Strands/AWS AgentCore deployment or runtime smoke path.",
    },
    "google_adk_eval_deployment": {
        "backend": "google_adk",
        "required_env": [
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ],
        "notes": "Live Google ADK eval execution and Vertex/Agent Engine deployment smoke path.",
    },
}


def build_smoke_matrix(*, environ: dict[str, str] | None = None) -> dict[str, object]:
    env = environ if environ is not None else os.environ
    enabled = env.get("AGENTBRIDGE_RUN_CREDENTIAL_SMOKE") == "1"
    return {
        "policy": (
            "Credentialed smoke checks are double gated: set "
            "AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 and the route-specific credentials."
        ),
        "enabled": enabled,
        "routes": {
            name: _route_status(name, config, env=env, enabled=enabled)
            for name, config in SMOKE_ROUTES.items()
        },
        "native_runtime_smokes": {
            name: _native_runtime_status(name, config, env=env, enabled=enabled)
            for name, config in NATIVE_RUNTIME_SMOKES.items()
        },
    }


def _route_status(
    name: str,
    config: dict[str, object],
    *,
    env: dict[str, str],
    enabled: bool,
) -> dict[str, object]:
    required_env = list(config["required_env"])
    missing_env = [key for key in required_env if not env.get(key)]
    for aliases in config.get("any_of_env", []):
        if not any(env.get(alias) for alias in aliases):
            missing_env.append(" or ".join(aliases))
    route = MODEL_ROUTES[name]
    return {
        "backend": config["backend"],
        "model": route["model"],
        "required_env": required_env,
        "any_of_env": config.get("any_of_env", []),
        "optional_env": config.get("optional_env", []),
        "missing_env": missing_env,
        "ready": enabled and not missing_env,
        "status": "ready" if enabled and not missing_env else "skipped",
        "notes": config["notes"],
    }


def _native_runtime_status(
    name: str,
    config: dict[str, object],
    *,
    env: dict[str, str],
    enabled: bool,
) -> dict[str, object]:
    del name
    required_env = list(config["required_env"])
    missing_env = [key for key in required_env if not env.get(key)]
    return {
        "backend": config["backend"],
        "required_env": required_env,
        "missing_env": missing_env,
        "ready": enabled and not missing_env,
        "status": "ready" if enabled and not missing_env else "skipped",
        "notes": config["notes"],
    }


if __name__ == "__main__":
    print(json.dumps(build_smoke_matrix(), indent=2, sort_keys=True))
