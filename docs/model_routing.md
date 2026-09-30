# Model Routing

AgentBridge standardizes the app-to-agent-framework layer. It does not try to become a second
model gateway. In v0, model selection should stay in:

- `AgentSpec.model`, using LiteLLM-style or backend-supported model strings.
- Backend-specific extension config for native model settings.
- Environment variables or SDK/client settings for API keys, base URLs, and provider credentials.

## Common Routes

| Route | Example `AgentSpec.model` | Extra Settings |
| --- | --- | --- |
| Offline CI | `agentbridge/offline` | No credentials; adapter-owned test model when supported. |
| OpenAI API | `openai/gpt-5` | `OPENAI_API_KEY` or backend-native auth. |
| Anthropic API | `anthropic/claude-sonnet` | `ANTHROPIC_API_KEY` or LiteLLM/provider settings. |
| Google API | `google/gemini` | Google credentials or ADK-native model settings. |
| Local Ollama-style model | `ollama/llama3.1` | Local model server running; backend must support that route. |
| OpenRouter | `openrouter/openai/gpt-4o-mini` | `OPENROUTER_API_KEY`, or OpenAI-compatible `base_url` in backend-native settings. |
| NVIDIA NIM | `openai/deepseek-ai/deepseek-v4-flash-0731` | `NVIDIA_NIM_API_KEY`, `NVIDIA_NIM_API_BASE` or `NVIDIA_NIM_BASE_URL`, optional `NVIDIA_MODEL`. |
| Custom gateway | `openai/internal-agent-model` | Internal OpenAI-compatible base URL and API key. |

## Backend Responsibilities

Adapters should:

- Pass `AgentSpec.model` through without rewriting provider semantics unless the backend requires it.
- Document which model string formats the backend has actually tested.
- Preserve native model settings in extension summaries when supplied.
- Keep credentialed provider tests optional and gated behind environment variables.
- Use `agentbridge/offline` for default CI whenever possible.

Adapters should not:

- Hide provider-specific model behavior behind fake portability.
- Require paid provider credentials for the default test suite.
- Claim that OpenAI-compatible endpoints work unless a smoke test or documented manual run exists.

For the LangChain plugin, an OpenAI-compatible native model can be configured without changing the
framework-neutral `AgentSpec` shape:

```python
agent = LangChainExtension.with_config(
    agent,
    model_provider="openai",
    model_options={
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_NIM_API_KEY"],
    },
)
```

Install `agentbridge-langchain[openai]` for this path. The adapter constructs the native
`ChatOpenAI` object, passes it to LangChain, and redacts secrets from `RunResult.metadata`.

## Executable Provider Smoke Matrix

The repository includes a double-gated smoke matrix for NVIDIA NIM, OpenRouter, and local Ollama:

```bash
AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 python examples/live_model_smoke.py
```

The callable form is `run_provider_smoke_matrix_from_env()`. It never runs by default. Configure
`NVIDIA_NIM_API_KEY` plus `NVIDIA_NIM_API_BASE` or `NVIDIA_NIM_BASE_URL` for NIM,
`OPENROUTER_API_KEY` for OpenRouter, or a reachable Ollama server. Optional overrides are
`NVIDIA_MODEL`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_MODEL`.
Each result records normalized content, usage, and provider errors without exposing API keys.

Observability providers use a separate executable lane:

```bash
AGENTBRIDGE_RUN_CREDENTIAL_SMOKE=1 \
  LANGSMITH_API_KEY=... \
  LANGFUSE_PUBLIC_KEY=... \
  LANGFUSE_SECRET_KEY=... \
  python examples/observability_smoke.py
```

This authenticates to LangSmith's `/info` endpoint and runs an offline LangChain invocation through
the Langfuse callback path. It does not claim hosted evaluation, prompt-management, or deployment
parity; those remain separate LangSmith API contract surfaces.

## OpenAI-Compatible Providers

Many providers expose OpenAI-compatible chat/completions APIs. AgentBridge should represent these
as a model string plus backend-native endpoint settings.

Example report shape:

```json
{
  "model": "openai/deepseek-ai/deepseek-v4-flash-0731",
  "provider": "nvidia_nim",
  "base_url_env": "NVIDIA_NIM_API_BASE",
  "alternate_base_url_env": "NVIDIA_NIM_BASE_URL",
  "api_key_env": "NVIDIA_NIM_API_KEY",
  "model_env": "NVIDIA_MODEL",
  "credentialed": true,
  "default_ci": false
}
```

This keeps the model routing story explicit without adding another abstraction layer.

Provider smoke outputs can be compared without exposing credentials:

```python
from examples.provider_comparison_report import build_provider_comparison_report

report = build_provider_comparison_report(route_results)
```

The report is result-driven and offline, making it suitable for CI artifacts and migration comparisons across NVIDIA NIM, OpenRouter, Ollama, and other compatible routes.

## Scenario Report Requirement

Deep scenario reports should include a `model_routes` section showing:

- The offline route used by CI.
- Hosted provider routes that are documented or tested.
- Local model routes.
- OpenAI-compatible endpoint routes such as OpenRouter, NVIDIA NIM, and internal gateways.
- Which routes are metadata-only, no-key runnable, or credential-gated smoke tests.
