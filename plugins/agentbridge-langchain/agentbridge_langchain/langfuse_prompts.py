"""Framework-neutral Langfuse prompt retrieval and compilation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .langfuse_api import LangfuseAPIClient


_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


@dataclass(frozen=True)
class LangfusePrompt:
    """A Langfuse prompt that can compile text or chat message variables."""

    name: str
    prompt: Any
    prompt_type: str | None = None
    version: int | None = None
    labels: tuple[str, ...] = ()
    config: Mapping[str, Any] | None = None
    raw: Any = None

    @classmethod
    def from_response(cls, name: str, response: Any) -> "LangfusePrompt":
        payload = response.get("data", response) if isinstance(response, dict) else response
        if not isinstance(payload, dict):
            raise TypeError("Langfuse prompt response must be an object.")
        return cls(
            name=str(payload.get("name", name)),
            prompt=payload.get("prompt", ""),
            prompt_type=payload.get("type"),
            version=payload.get("version"),
            labels=tuple(payload.get("labels") or ()),
            config=payload.get("config"),
            raw=response,
        )

    def compile(self, **variables: Any) -> Any:
        """Replace Langfuse ``{{variable}}`` placeholders in text or chat prompts."""

        return _compile_value(self.prompt, variables)

    def get_langchain_prompt(self, **variables: Any) -> Any:
        """Return a LangChain-compatible template with remaining variables as ``{name}``."""

        compiled = self.compile(**variables)
        return _langchain_template(compiled)


def fetch_prompt(
    client: LangfuseAPIClient,
    name: str,
    *,
    label: str | None = None,
    version: int | None = None,
    prompt_type: str | None = None,
) -> LangfusePrompt:
    """Fetch and wrap a versioned or labeled Langfuse prompt."""

    response = client.get_prompt(
        name,
        label=label,
        version=version,
        prompt_type=prompt_type,
    )
    return LangfusePrompt.from_response(name, response)


async def afetch_prompt(
    client: LangfuseAPIClient,
    name: str,
    *,
    label: str | None = None,
    version: int | None = None,
    prompt_type: str | None = None,
) -> LangfusePrompt:
    """Async fetch and wrap a versioned or labeled Langfuse prompt."""

    response = await client.aget_prompt(
        name,
        label=label,
        version=version,
        prompt_type=prompt_type,
    )
    return LangfusePrompt.from_response(name, response)


def _compile_value(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _VARIABLE.sub(
            lambda match: str(variables.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [_compile_value(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(_compile_value(item, variables) for item in value)
    if isinstance(value, dict):
        return {key: _compile_value(item, variables) for key, item in value.items()}
    return value


def _langchain_template(value: Any) -> Any:
    if isinstance(value, str):
        return _VARIABLE.sub(lambda match: "{" + match.group(1) + "}", value)
    if isinstance(value, list):
        return [_langchain_template(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_langchain_template(item) for item in value)
    if isinstance(value, dict):
        return {key: _langchain_template(item) for key, item in value.items()}
    return value
