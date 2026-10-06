"""Exercise a Langfuse review queue with an injected, stateful no-key transport."""

import asyncio
import json
from urllib.parse import parse_qs, urlparse

from agentbridge_langchain import LangfuseAPIClient


def offline_client():
    config, queue, items, assignments = {}, {}, {}, set()

    def transport(method, url, headers, raw):
        parsed = urlparse(url)
        path = parsed.path.removeprefix("/api/public/")
        query = parse_qs(parsed.query)
        body = json.loads(raw) if raw else {}
        if path == "score-configs" and method == "POST":
            config.update(id="config-1", isArchived=False, **body)
            response = dict(config)
        elif path == "score-configs/config-1":
            if method == "PATCH":
                config.update(body)
            response = dict(config)
        elif path == "annotation-queues" and method == "POST":
            queue.update(id="queue-1", **body)
            response = dict(queue)
        elif path == "annotation-queues/queue-1":
            response = dict(queue)
        elif path == "annotation-queues/queue-1/assignments":
            if method == "POST":
                assignments.add(body["userId"])
            else:
                assignments.discard(body["userId"])
            response = {"success": True}
        elif path == "annotation-queues/queue-1/items" and method == "POST":
            item = {"id": f"item-{len(items) + 1}", "status": "PENDING", **body}
            items[item["id"]] = item
            response = dict(item)
        elif path.startswith("annotation-queues/queue-1/items/"):
            item_id = path.rsplit("/", 1)[-1]
            if method == "DELETE":
                del items[item_id]
                response = {"success": True}
            else:
                if method == "PATCH":
                    items[item_id].update(body)
                response = dict(items[item_id])
        elif method == "GET" and path in ("score-configs", "annotation-queues", "annotation-queues/queue-1/items"):
            records = ([dict(config)] if path == "score-configs" else [dict(queue)]
                       if path == "annotation-queues" else list(items.values()))
            status = query.get("status", [None])[0]
            if status:
                records = [record for record in records if record["status"] == status]
            page = int(query.get("page", ["1"])[0])
            limit = int(query.get("limit", ["50"])[0])
            response = {"data": records[(page - 1) * limit:page * limit],
                        "meta": {"page": page, "limit": limit, "totalItems": len(records),
                                 "totalPages": (len(records) + limit - 1) // limit}}
        elif path == "scores" and method == "POST":
            response = {"id": "score-1", **body}
        else:
            return 404, {"content-type": "application/json"}, b'{"error":"unsupported offline route"}'
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    return LangfuseAPIClient(public_key="offline", secret_key="offline", transport=transport)


async def workflow(client, *, asynchronous):
    async def call(name, /, *args, **kwargs):
        if asynchronous:
            return await getattr(client, "a" + name)(*args, **kwargs)
        return getattr(client, name)(*args, **kwargs)

    async def export_items(queue_id, **query):
        if asynchronous:
            return [item async for item in client.aiter_annotation_queue_items(queue_id, query=query)]
        return list(client.iter_annotation_queue_items(queue_id, query=query))

    config = await call("create_score_config", name="refund_approved", data_type="BOOLEAN")
    queue = await call("create_annotation_queue", name="refund_review", score_config_ids=[config["id"]])
    await call("create_annotation_queue_assignment", queue["id"], user_id="existing-reviewer")
    item = await call("create_annotation_queue_item", queue["id"], object_id="existing-trace", object_type="TRACE")
    pending = await export_items(queue["id"], status="PENDING")
    # This is an explicit fixture review decision, not an automated human approval.
    score = await call("create_score", trace_id=item["objectId"], name="refund_approved",
                       value=False, data_type="BOOLEAN", config_id=config["id"],
                       comment="Reviewer rejected the refund.")
    await call("update_annotation_queue_item", queue["id"], item["id"], body={"status": "COMPLETED"})
    completed = await export_items(queue["id"], status="COMPLETED")
    await call("delete_annotation_queue_assignment", queue["id"], user_id="existing-reviewer")
    await call("delete_annotation_queue_item", queue["id"], item["id"])
    archived = await call("update_score_config", config["id"], body={"isArchived": True})
    return {"pending": pending, "completed": completed, "score": score, "config": archived,
            "remaining_items": await export_items(queue["id"])}


def run_demo():
    synchronous = asyncio.run(workflow(offline_client(), asynchronous=False))
    asynchronous = asyncio.run(workflow(offline_client(), asynchronous=True))
    return {"execution": "stateful injected transport, no hosted API calls",
            "sync": synchronous, "async": asynchronous, "parity": synchronous == asynchronous}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
