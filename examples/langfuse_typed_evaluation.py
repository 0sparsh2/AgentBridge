"""Publish and export typed evaluation scores with a deterministic no-key transport."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

from agentbridge import EvaluationCase, EvaluationExample, EvaluationReport, EvaluationScore
from agentbridge_langchain import LangfuseAPIClient, apublish_report_scores, publish_report_scores


def build_report():
    return EvaluationReport(dataset_name="refunds", backend="mock", cases=[
        EvaluationCase(
            index=0, example=EvaluationExample(input="Check order A123"),
            result={"eligible": False}, backend="mock", scores=[
                EvaluationScore(key="quality", score=0.95),
                EvaluationScore(key="approved", value=False),
                EvaluationScore(key="verdict", value="reject", metadata={"policy": "v2"}),
                EvaluationScore(key="explanation", value="Outside the refund window."),
            ],
        ),
    ])


def run_demo():
    records = []

    def transport(method, url, headers, body):
        if method == "POST":
            payload = json.loads(body)
            record = {
                "id": f"score-{len(records)}", "name": payload["name"],
                "value": payload["value"], "dataType": payload["dataType"],
                "subject": {"kind": "trace", "id": payload["traceId"]},
            }
            if "metadata" in payload:
                record["metadata"] = payload["metadata"]
            records.append(record)
            response = {"id": record["id"]}
        else:
            query = parse_qs(urlparse(url).query)
            offset = int(query.get("cursor", ["0"])[0])
            response = {
                "data": records[offset:offset + 2],
                "meta": {"cursor": str(offset + 2) if offset + 2 < len(records) else None},
            }
        return 200, {"content-type": "application/json"}, json.dumps(response).encode()

    client = LangfuseAPIClient(public_key="offline", secret_key="offline", transport=transport)
    report = build_report()
    score_types = {"verdict": "CATEGORICAL"}
    publish_report_scores(report, trace_ids={0: "trace-demo"}, client=client, score_types=score_types)
    synchronous = list(client.iter_scores_v3(query={"fields": "details,subject"}))
    records.clear()

    async def collect():
        await apublish_report_scores(
            report, trace_ids={0: "trace-demo"}, client=client, score_types=score_types
        )
        return [item async for item in client.aiter_scores_v3(query={"fields": "details,subject"})]

    asynchronous = asyncio.run(collect())
    return {"execution": "injected transport, no hosted API calls",
            "sync": synchronous, "async": asynchronous, "parity": synchronous == asynchronous}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
