"""Decode complete SSE responses while retaining transport event metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def parse_sse(raw: bytes) -> Iterator[dict[str, Any]]:
    """Preserve named events; keep data-only JSON envelopes backward compatible."""

    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    for line in [*raw.decode("utf-8-sig").splitlines(), ""]:
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                if data == "[DONE]":
                    return
                payload = json.loads(data)
                if event_name is not None or event_id is not None:
                    envelope: dict[str, Any] = {"data": payload}
                    if event_name is not None:
                        envelope["event"] = event_name
                    if event_id is not None:
                        envelope["id"] = event_id
                    yield envelope
                elif isinstance(payload, dict):
                    yield payload
                else:
                    yield {"data": payload}
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or None
        elif field == "id" and "\x00" not in value:
            event_id = value
        elif field == "data":
            data_lines.append(value)
