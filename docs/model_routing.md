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

## Scenario Report Requirement

Deep scenario reports should include a `model_routes` section showing:

- The offline route used by CI.
- Hosted provider routes that are documented or tested.
- Local model routes.
- OpenAI-compatible endpoint routes such as OpenRouter, NVIDIA NIM, and internal gateways.
- Which routes are metadata-only, no-key runnable, or credential-gated smoke tests.
