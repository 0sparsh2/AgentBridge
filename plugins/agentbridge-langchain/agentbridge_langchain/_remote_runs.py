"""Native Agent Server run payload construction."""

from collections.abc import Mapping
from typing import Any


def run_body(
    assistant_id: str,
    input: Any,
    *,
    command: Mapping[str, Any] | None = None,
    run_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve server options without overriding explicitly supplied run identity/input."""

    options = dict(run_options or {})
    reserved = {"assistant_id", "input", "command"}.intersection(options)
    if reserved:
        raise ValueError(f"run_options cannot override reserved fields: {', '.join(sorted(reserved))}")
    body = {"assistant_id": assistant_id, "input": input, **options}
    if command is not None:
        body["command"] = dict(command)
    return body
