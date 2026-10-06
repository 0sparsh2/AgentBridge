import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest

from agentbridge_langchain import LangfuseAPIClient


CONFIG_ID = "config/a?b#c"
QUEUE_ID = "queue/a?b#c"
ITEM_ID = "item/a?b#c"
CONFIG_PATH = "/api/public/score-configs/config%2Fa%3Fb%23c"
QUEUE_PATH = "/api/public/annotation-queues/queue%2Fa%3Fb%23c"
ITEM_PATH = QUEUE_PATH + "/items/item%2Fa%3Fb%23c"

OPERATIONS = [
    ("list_score_configs", (), {"query": {"page": 2, "limit": 5}}, "GET", "/api/public/score-configs?page=2&limit=5", None),
    ("get_score_config", (CONFIG_ID,), {}, "GET", CONFIG_PATH, None),
    ("create_score_config", (), {"name": "quality", "data_type": "NUMERIC", "min_value": 0,
                                 "max_value": 1, "description": ""}, "POST", "/api/public/score-configs",
     {"name": "quality", "dataType": "NUMERIC", "minValue": 0, "maxValue": 1, "description": ""}),
    ("update_score_config", (CONFIG_ID,), {"body": {"isArchived": False, "description": None}},
     "PATCH", CONFIG_PATH, {"isArchived": False, "description": None}),
    ("list_annotation_queues", (), {"query": {"limit": 5}}, "GET", "/api/public/annotation-queues?limit=5", None),
    ("get_annotation_queue", (QUEUE_ID,), {}, "GET", QUEUE_PATH, None),
    ("create_annotation_queue", (), {"name": "refunds", "score_config_ids": ["config-1"], "description": ""},
     "POST", "/api/public/annotation-queues", {"name": "refunds", "scoreConfigIds": ["config-1"], "description": ""}),
    ("list_annotation_queue_items", (QUEUE_ID,), {"query": {"status": "PENDING", "page": 2}},
     "GET", QUEUE_PATH + "/items?status=PENDING&page=2", None),
    ("get_annotation_queue_item", (QUEUE_ID, ITEM_ID), {}, "GET", ITEM_PATH, None),
    ("create_annotation_queue_item", (QUEUE_ID,), {"object_id": "trace-1", "object_type": "TRACE", "status": "PENDING"},
     "POST", QUEUE_PATH + "/items", {"objectId": "trace-1", "objectType": "TRACE", "status": "PENDING"}),
    ("update_annotation_queue_item", (QUEUE_ID, ITEM_ID), {"body": {"status": "COMPLETED"}},
     "PATCH", ITEM_PATH, {"status": "COMPLETED"}),
    ("delete_annotation_queue_item", (QUEUE_ID, ITEM_ID), {}, "DELETE", ITEM_PATH, None),
    ("create_annotation_queue_assignment", (QUEUE_ID,), {"user_id": "reviewer-1"},
     "POST", QUEUE_PATH + "/assignments", {"userId": "reviewer-1"}),
    ("delete_annotation_queue_assignment", (QUEUE_ID,), {"user_id": "reviewer-1"},
     "DELETE", QUEUE_PATH + "/assignments", {"userId": "reviewer-1"}),
]


@pytest.mark.parametrize("operation", OPERATIONS, ids=[item[0] for item in OPERATIONS])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_governance_helpers_preserve_native_http_contracts(operation, asynchronous):
    name, args, kwargs, method, path, body = operation
    calls = []

    def transport(verb, url, headers, payload):
        calls.append((verb, url, json.loads(payload) if payload else None))
        return 200, {"content-type": "application/json"}, b'{"id":"native","extra":{"preserved":true}}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    if asynchronous:
        result = asyncio.run(getattr(client, "a" + name)(*args, **kwargs))
    else:
        result = getattr(client, name)(*args, **kwargs)
    assert result == {"id": "native", "extra": {"preserved": True}}
    assert calls == [(method, "https://cloud.langfuse.com" + path, body)]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_categorical_config_preserves_native_categories_and_omits_unspecified_options(asynchronous):
    bodies = []
    categories = [{"label": "reject", "value": 0}, {"label": "approve", "value": 1}]

    def transport(method, url, headers, body):
        bodies.append(json.loads(body))
        return 200, {"content-type": "application/json"}, b'{"id":"config-1"}'

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    kwargs = {"name": "verdict", "data_type": "CATEGORICAL", "categories": categories}
    if asynchronous:
        asyncio.run(client.acreate_score_config(**kwargs))
    else:
        client.create_score_config(**kwargs)
    assert bodies == [{"name": "verdict", "dataType": "CATEGORICAL", "categories": categories}]
    assert categories == [{"label": "reject", "value": 0}, {"label": "approve", "value": 1}]


@pytest.mark.parametrize("resource", ["score_configs", "annotation_queues", "annotation_queue_items"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_governance_iterators_follow_numbered_pages_and_preserve_status_filters(resource, asynchronous):
    calls = []
    filters = {"limit": 1}
    args = ()
    if resource == "annotation_queue_items":
        args = (QUEUE_ID,)
        filters["status"] = "PENDING"

    def transport(method, url, headers, body):
        query = parse_qs(urlparse(url).query)
        calls.append((urlparse(url).path, query))
        page = int(query.get("page", ["1"])[0])
        response = {"data": [{"id": f"record-{page}"}],
                    "meta": {"page": page, "limit": 1, "totalItems": 2, "totalPages": 2}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    if asynchronous:
        async def collect():
            return [item async for item in getattr(client, "aiter_" + resource)(*args, query=filters)]

        result = asyncio.run(collect())
    else:
        result = list(getattr(client, "iter_" + resource)(*args, query=filters))
    assert result == [{"id": "record-1"}, {"id": "record-2"}]
    assert len(calls) == 2
    assert calls[1][1]["page"] == ["2"]
    assert "page" not in filters
    if args:
        assert all(path == QUEUE_PATH + "/items" and query["status"] == ["PENDING"] for path, query in calls)


@pytest.mark.parametrize("identifier", ["", ".", "..", None])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_governance_helpers_reject_invalid_resource_identifiers_before_requests(identifier, asynchronous):
    def transport(*args):
        pytest.fail("Invalid identifiers must not make requests")

    client = LangfuseAPIClient(public_key="pk", secret_key="sk", transport=transport)
    with pytest.raises(ValueError, match="identifier"):
        if asynchronous:
            asyncio.run(client.aget_annotation_queue_item("queue-1", identifier))
        else:
            client.get_annotation_queue_item("queue-1", identifier)


def test_governance_workflow_example_preserves_sync_async_state_transitions():
    from examples.langfuse_review_queue import run_demo

    result = run_demo()
    assert result["parity"] is True
    assert result["sync"]["completed"][0]["status"] == "COMPLETED"
    assert result["sync"]["score"]["value"] is False
    assert result["sync"]["config"]["isArchived"] is True
    assert result["sync"]["remaining_items"] == []
