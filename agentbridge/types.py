"""Framework-neutral AgentBridge data types."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

EventType = Literal["message", "tool_call", "tool_result", "error", "complete"]
CapabilityStatus = Literal["full", "partial", "extension", "native_only", "unsupported"]


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    if isinstance(annotation, str):
        normalized = annotation.lower()
        if normalized in {"str", "string"}:
            return {"type": "string"}
        if normalized in {"int", "integer"}:
            return {"type": "integer"}
        if normalized in {"float", "number"}:
            return {"type": "number"}
        if normalized in {"bool", "boolean"}:
            return {"type": "boolean"}
        if normalized in {"dict", "mapping"}:
            return {"type": "object"}
        if normalized.startswith("list"):
            return {"type": "array"}

    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return {"type": "array"}
    if annotation in (str,):
        return {"type": "string"}
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation in (bool,):
        return {"type": "boolean"}
    if annotation in (dict,):
        return {"type": "object"}
    return {"type": "string"}


class ToolSpec(BaseModel):
    """Framework-neutral Python tool definition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any] = Field(exclude=True, repr=False)

    @classmethod
    def from_function(
        cls,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> "ToolSpec":
        """Create a tool spec from a Python callable."""

        signature = inspect.signature(function)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for parameter_name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            properties[parameter_name] = _annotation_to_schema(parameter.annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        doc = inspect.getdoc(function) or f"Run {function.__name__}."
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return cls(
            name=name or function.__name__,
            description=description or doc,
            input_schema=schema,
            handler=function,
        )

    def call(self, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke the wrapped callable with keyword arguments."""

        return self.handler(**(arguments or {}))


class AgentSpec(BaseModel):
    """A framework-neutral definition of an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    output_type: Any | None = Field(default=None, exclude=True, repr=False)
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    backend_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "instructions", "model")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @model_validator(mode="after")
    def _derive_output_schema(self) -> "AgentSpec":
        if self.output_schema is None and hasattr(self.output_type, "model_json_schema"):
            self.output_schema = self.output_type.model_json_schema()
        return self


class BackendCapabilities(BaseModel):
    """Feature coverage advertised by a backend adapter."""

    backend: str = Field(min_length=1)
    features: dict[str, CapabilityStatus] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)

    def status(self, feature: str) -> CapabilityStatus:
        """Return the support status for a feature."""

        return self.features.get(feature, "unsupported")

    def supports(self, feature: str) -> bool:
        """Return whether a feature is usable through AgentBridge."""

        return self.status(feature) in {"full", "partial", "extension"}


class RunInput(BaseModel):
    """Input for a single agent run."""

    input: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None

    @field_validator("input")
    @classmethod
    def _input_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input cannot be blank")
        return value


class AgentEvent(BaseModel):
    """A normalized agent stream event."""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
    backend: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """A normalized result returned by any backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any
    backend: str
    events: list[AgentEvent] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: Any = Field(default=None, exclude=True, repr=False)
    _raw: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._raw = self.raw
