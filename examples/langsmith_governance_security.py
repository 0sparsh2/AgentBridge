"""Show safe LangSmith governance URL construction without contacting a hosted service."""

import asyncio
import json
from urllib.parse import urlparse

from agentbridge_langchain import LangSmithAPIClient


def run_demo():
    calls = []

    def transport(method, url, headers, body):
        del headers
        calls.append({
            "method": method,
            "path": urlparse(url).path,
            "body": json.loads(body) if body else None,
        })
        status = 204 if method == "DELETE" else 200
        return status, {"content-type": "application/json"}, b"" if status == 204 else b'{"ok":true}'

    def client():
        return LangSmithAPIClient(api_key="offline", base_url="https://example.test", transport=transport)

    async def workflow(api, *, asynchronous):
        async def call(method, /, *args, **kwargs):
            if asynchronous:
                return await getattr(api, "a" + method)(*args, **kwargs)
            return getattr(api, method)(*args, **kwargs)

        agent_id = "agent/a?b#c"
        queue_id = "queue/a?b#c"
        run_id = "run/a?b#c"
        await call("create_agent_connection", agent_id, body={"provider": "github"})
        await call("add_runs_to_annotation_queue", queue_id, run_ids=[run_id])
        await call("remove_run_from_annotation_queue", queue_id, run_id)
        await call("revoke_connection_token", "token/a?b#c")
        return [call["path"] for call in calls]

    synchronous = asyncio.run(workflow(client(), asynchronous=False))
    calls.clear()
    asynchronous = asyncio.run(workflow(client(), asynchronous=True))
    return {
        "execution": "injected transport, no hosted API calls",
        "sync": synchronous,
        "async": asynchronous,
        "parity": synchronous == asynchronous,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
