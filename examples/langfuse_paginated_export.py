"""Export prompts and evaluation datasets using native numbered pagination, without keys."""

import asyncio
import json
from urllib.parse import parse_qs, urlparse

from agentbridge_langchain import LangfuseAPIClient


def run_demo():
    def transport(method, url, headers, body):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        page = int(query.get("page", ["1"])[0])
        records = [{"id": f"{parsed.path.rsplit('/', 1)[-1]}-{index}"} for index in range(3)]
        response = {"data": records[(page - 1) * 2:page * 2],
                    "meta": {"page": page, "limit": 2, "totalItems": 3, "totalPages": 2}}
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="offline", secret_key="offline", transport=transport)
    queries = {"prompts": {"label": "production", "limit": 2},
               "datasets": {"limit": 2}, "dataset_items": {"datasetName": "refunds", "limit": 2}}
    synchronous = {name: list(getattr(client, f"iter_{name}")(query=query))
                   for name, query in queries.items()}

    async def export():
        results = {}
        for name, query in queries.items():
            results[name] = [item async for item in getattr(client, f"aiter_{name}")(query=query)]
        return results

    asynchronous = asyncio.run(export())
    return {"execution": "injected transport, no hosted API calls", "sync": synchronous,
            "async": asynchronous, "parity": synchronous == asynchronous}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
