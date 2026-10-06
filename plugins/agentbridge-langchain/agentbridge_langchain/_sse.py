"""Incremental SSE decoding and dependency-free HTTP streaming."""

from __future__ import annotations

import json
import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from itertools import chain
from typing import Any, TypeVar
from urllib.request import Request, urlopen


def parse_sse(raw: bytes) -> Iterator[dict[str, Any]]:
    """Preserve named events; keep data-only JSON envelopes backward compatible."""

    yield from parse_sse_lines(raw.decode("utf-8-sig").splitlines())


def parse_sse_lines(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    """Decode each completed frame without consuming later response lines."""

    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    first_line = True
    for line in chain(lines, ("",)):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        line = line.rstrip("\r\n")
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


def stream_http(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, Mapping[str, str], Iterator[bytes]]:
    """Open an HTTP response and release it when its iterator is closed."""

    response = urlopen(Request(url, data=body, headers=headers, method=method), timeout=30)

    def lines() -> Iterator[bytes]:
        try:
            yield from response
        finally:
            response.close()

    return response.status, dict(response.headers.items()), lines()


def response_events(raw: bytes | Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Decode buffered custom transports or incremental HTTP responses."""

    if isinstance(raw, bytes):
        yield from parse_sse(raw)
        return
    try:
        yield from parse_sse_lines(raw)
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


EventT = TypeVar("EventT")


async def async_events(events: Iterator[EventT]) -> AsyncIterator[EventT]:
    """Read one frame at a time off-loop and close after a read finishes."""

    exhausted = object()
    try:
        while True:
            pending = asyncio.create_task(asyncio.to_thread(next, events, exhausted))
            try:
                event = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A Python generator cannot be closed while its worker is executing next().
                try:
                    await pending
                finally:
                    raise
            if event is exhausted:
                return
            yield event
    finally:
        await asyncio.to_thread(events.close)
