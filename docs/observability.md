# Observability Correlation

AgentBridge keeps LangSmith and Langfuse as optional integrations, but uses one metadata contract
for LangChain and LangGraph runs. `observability_metadata()` merges application metadata with
session, user, project, and trace identifiers without storing provider secrets in `AgentSpec`.

```python
from agentbridge.observability import observability_metadata

metadata = observability_metadata(
    {"observability": {
        "langsmith": {"enabled": True, "project_name": "refunds"},
        "langfuse": {"enabled": True, "user_id": "customer-123"},
    }},
    session_id="session-1",
)
```

The same metadata is forwarded into LangChain and LangGraph runtime configuration. LangSmith
tracing context and Langfuse callbacks remain lazy and optional; credentialed API smoke tests are
separate from the credential-free correlation contract.

## Langfuse Telemetry API

The LangChain plugin exposes the current OTLP/HTTP path and retains the legacy JSON ingestion
helper only as a compatibility escape hatch:

```python
from agentbridge_langchain.langfuse_api import LangfuseAPIClient

client = LangfuseAPIClient()
client.ingest_otlp({"resourceSpans": []})
client.list_observations(query={"limit": 100})
client.list_scores_v3(query={"dataType": "NUMERIC"})
list(client.iter_observations(query={"traceId": "trace-123", "limit": 100}))
list(client.iter_scores_v3(query={"dataType": "NUMERIC", "limit": 100}))
client.query_metrics(query={"view": "traces"})
client.list_experiments(query={"fields": "core,scores"})
client.list_experiment_items(query={"experimentId": "exp-123", "fields": "io,scores"})
client.list_datasets(query={"limit": 50})
client.list_dataset_items(query={"datasetName": "refunds"})
```

`ingest_otlp()` also accepts protobuf bytes with `content_type="application/x-protobuf"` and sends
`x-langfuse-ingestion-version: 4` by default. The iterators follow `meta.cursor` across pages while
preserving the caller's filters. The generic request methods remain available for new Langfuse
endpoints without waiting for a core release.

Dataset lifecycle helpers use the current v2 dataset routes and the versioned dataset-item API:
`create_dataset`, `list_datasets`, `get_dataset`, `delete_dataset`, and item create/upsert/list/get/
delete operations. Legacy trace reads remain available only as a compatibility escape hatch; new
trace extraction should query Observations v2 with a `traceId` filter. Trace deletion helpers are
also available for cleanup workflows.

## Langfuse Prompts

Prompt versions and deployment labels can be fetched without coupling AgentBridge core to the
Langfuse SDK:

```python
from agentbridge_langchain.langfuse_prompts import fetch_prompt

prompt = fetch_prompt(client, "refund-policy", label="production")
text = prompt.compile(customer_tier="gold")
chat_messages = prompt.get_langchain_prompt()
```

The wrapper preserves the raw response, supports text and chat prompts, and leaves unresolved
Langfuse variables as LangChain-style `{name}` placeholders.
