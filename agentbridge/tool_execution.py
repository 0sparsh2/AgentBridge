"""Shared helpers for adapter tool execution."""

from __future__ import annotations

from typing import Any

from agentbridge.types import RunInput, ToolSpec


def arguments_from_schema(schema: dict[str, Any], run_input: RunInput) -> dict[str, Any]:
    """Build deterministic tool arguments from a simple JSON-schema-like shape."""

    properties = schema.get("properties", {})
    arguments: dict[str, Any] = {}
    for name, property_schema in properties.items():
        property_type = property_schema.get("type")
        if name in run_input.context:
            arguments[name] = run_input.context[name]
        elif property_type == "integer":
            arguments[name] = 1
        elif property_type == "number":
            arguments[name] = 1.0
        elif property_type == "boolean":
            arguments[name] = True
        elif property_type == "array":
            arguments[name] = []
        elif property_type == "object":
            arguments[name] = {}
        else:
            arguments[name] = run_input.input
    return arguments


def execute_sync_tools(tools: list[ToolSpec], run_input: RunInput) -> list[dict[str, Any]]:
    """Execute sync ToolSpecs and return normalized call/result records."""

    records: list[dict[str, Any]] = []
    for tool in tools:
        arguments = arguments_from_schema(tool.input_schema, run_input)
        result = tool.call(arguments)
        records.append({"name": tool.name, "arguments": arguments, "result": result})
    return records
