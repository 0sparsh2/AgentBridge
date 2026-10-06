import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from agentbridge_langchain import LangfuseAPIClient


RESOURCES = ("prompts", "datasets", "dataset_items")
CURSOR_RESOURCES = ("observations", "scores_v3", "experiments", "experiment_items")


def collect(client, resource, query, asynchronous):
    if not asynchronous:
        return list(getattr(client, f"iter_{resource}")(query=query))

    async def run():
        return [item async for item in getattr(client, f"aiter_{resource}")(query=query)]

    return asyncio.run(run())


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_numbered_exports_preserve_filters_start_page_and_native_metadata(resource, asynchronous):
    calls = []
    filters = {"page": 2, "limit": 1}
    if resource == "prompts":
        filters.update(label="production", tag="refunds")
    elif resource == "dataset_items":
        filters.update(datasetName="refunds", version="2026-01-21T14:35:42Z")
    original = dict(filters)

    def transport(method, url, headers, body):
        query = parse_qs(urlparse(url).query)
        calls.append(query)
        assert method == "GET"
        assert "cursor" not in query
        page = int(query["page"][0])
        response = {"data": [{"id": f"item-{page}"}],
                    "meta": {"page": page, "limit": 1, "totalItems": 3, "totalPages": 3}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert collect(client, resource, filters, asynchronous) == [{"id": "item-2"}, {"id": "item-3"}]
    assert filters == original
    assert [query["page"] for query in calls] == [["2"], ["3"]]
    for query in calls:
        for key, value in filters.items():
            if key != "page":
                assert query[key] == [str(value)]


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_numbered_exports_accept_empty_native_response(resource, asynchronous):
    calls = []

    def transport(method, url, headers, body):
        calls.append(url)
        response = {"data": [], "meta": {"page": 1, "limit": 50, "totalItems": 0, "totalPages": 0}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert collect(client, resource, None, asynchronous) == []
    assert len(calls) == 1


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("query", [{"cursor": "next"}, {"page": 0}, {"page": True}, {"page": "2"}])
def test_numbered_exports_reject_invalid_query_before_requests(resource, asynchronous, query):
    def transport(*args):
        pytest.fail("Invalid pagination must not make an HTTP request")

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    with pytest.raises(ValueError, match="page"):
        collect(client, resource, query, asynchronous)


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["repeated", "missing", "malformed", "bad_total"])
def test_numbered_exports_fail_explicitly_on_incomplete_responses(resource, asynchronous, failure):
    calls = []

    def transport(method, url, headers, body):
        page = int(parse_qs(urlparse(url).query).get("page", ["1"])[0])
        calls.append(page)
        response = {"data": [{"id": str(page)}],
                    "meta": {"page": page, "limit": 1, "totalItems": 2, "totalPages": 2}}
        if page == 2:
            if failure == "repeated":
                response["meta"]["page"] = 1
            elif failure == "missing":
                response["meta"] = {}
            elif failure == "malformed":
                response["data"] = [None]
            else:
                response["meta"]["totalPages"] = "2"
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    with pytest.raises(RuntimeError, match="export is incomplete"):
        collect(client, resource, None, asynchronous)
    assert calls == [1, 2]


def test_numbered_export_example_preserves_sync_async_parity():
    from examples.langfuse_paginated_export import run_demo

    result = run_demo()
    assert result["parity"] is True
    assert all(len(records) == 3 for records in result["sync"].values())


@pytest.mark.parametrize("resource", CURSOR_RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cursor_exports_preserve_opaque_cursors_filters_and_records(resource, asynchronous):
    calls = []
    filters = {"fromStartTime": "2026-01-01T00:00:00Z", "fields": "details,subject", "limit": 1}
    original = dict(filters)

    def transport(method, url, headers, body):
        query = parse_qs(urlparse(url).query)
        calls.append(query)
        second = "cursor" in query
        if second:
            assert query["cursor"] == ["opaque/+ ="]
        response = {"data": [{"id": "second" if second else "first", "value": False}],
                    "meta": {} if second else {"cursor": "opaque/+ ="}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    assert collect(client, resource, filters, asynchronous) == [
        {"id": "first", "value": False}, {"id": "second", "value": False},
    ]
    assert len(calls) == 2
    assert filters == original
    for query in calls:
        assert "page" not in query
        for key, value in filters.items():
            assert query[key] == [str(value)]


@pytest.mark.parametrize("resource", CURSOR_RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("initial_cursor", [None, "a"])
def test_cursor_exports_reject_longer_cycles_in_all_resources(resource, asynchronous, initial_cursor):
    calls = []

    def transport(method, url, headers, body):
        cursor = parse_qs(urlparse(url).query).get("cursor", [None])[0]
        calls.append(cursor)
        response = {"data": [{"id": str(cursor)}], "meta": {"cursor": {None: "a", "a": "b", "b": "a"}[cursor]}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    filters = {"cursor": initial_cursor} if initial_cursor else None
    with pytest.raises(RuntimeError, match="export is incomplete"):
        collect(client, resource, filters, asynchronous)
    assert calls == (["a", "b"] if initial_cursor else [None, "a", "b"])


@pytest.mark.parametrize("resource", CURSOR_RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response", [{"data": []}, {"data": [None], "meta": {}},
                                      {"data": [], "meta": {"cursor": []}}])
def test_cursor_exports_reject_malformed_native_responses(resource, asynchronous, response):
    def transport(method, url, headers, body):
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    with pytest.raises(RuntimeError, match="export is incomplete"):
        collect(client, resource, None, asynchronous)


@pytest.mark.parametrize("resource", CURSOR_RESOURCES)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("query", [{"cursor": []}, {"cursor": True}, {"page": 1}])
def test_cursor_exports_reject_invalid_queries_before_requests(resource, asynchronous, query):
    def transport(*args):
        pytest.fail("Invalid pagination must not make an HTTP request")

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    with pytest.raises(ValueError, match="cursor"):
        collect(client, resource, query, asynchronous)
