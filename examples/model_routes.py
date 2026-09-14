"""Print AgentBridge model-routing examples without contacting providers."""

from __future__ import annotations

import json


MODEL_ROUTES = {
    "offline_ci": {
        "model": "agentbridge/offline",
        "credentialed": False,
        "default_ci": True,
        "notes": "Adapter-owned offline model path for no-key conformance and examples.",
    },
    "openai_api": {
        "model": "openai/gpt-5",
        "credentialed": True,
        "api_key_env": "OPENAI_API_KEY",
        "default_ci": False,
    },
    "anthropic_api": {
        "model": "anthropic/claude-sonnet",
        "credentialed": True,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_ci": False,
    },
    "google_api": {
        "model": "google/gemini",
        "credentialed": True,
        "api_key_env": "GOOGLE_API_KEY",
        "default_ci": False,
    },
    "local_ollama": {
        "model": "ollama/llama3.1",
        "credentialed": False,
        "base_url_env": "OLLAMA_BASE_URL",
        "default_ci": False,
        "notes": "Requires a local Ollama-compatible server; not assumed in CI.",
    },
    "openrouter": {
        "model": "openrouter/openai/gpt-4o-mini",
        "credentialed": True,
        "api_key_env": "OPENROUTER_API_KEY",
        "default_ci": False,
    },
    "nvidia_nim_openai_compatible": {
        "model": "openai/deepseek-ai/deepseek-v4-flash-0731",
        "credentialed": True,
        "api_key_env": "NVIDIA_NIM_API_KEY",
        "base_url_env": "NVIDIA_NIM_API_BASE",
        "alternate_base_url_env": "NVIDIA_NIM_BASE_URL",
        "model_env": "NVIDIA_MODEL",
        "default_ci": False,
        "notes": "Represent NIM through an OpenAI-compatible route plus backend-native base URL.",
    },
    "custom_openai_compatible_gateway": {
        "model": "openai/internal-agent-model",
        "credentialed": True,
        "api_key_env": "AGENTBRIDGE_GATEWAY_API_KEY",
        "base_url_env": "AGENTBRIDGE_GATEWAY_BASE_URL",
        "default_ci": False,
    },
}


def build_model_route_catalog() -> dict[str, object]:
    return {
        "policy": "AgentBridge records model routes but delegates provider calls to LiteLLM-style strings or backend-native settings.",
        "routes": MODEL_ROUTES,
    }


if __name__ == "__main__":
    print(json.dumps(build_model_route_catalog(), indent=2, sort_keys=True))
