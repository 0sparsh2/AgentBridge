# Evaluation

AgentBridge provides a small, backend-neutral evaluation contract for local
regression tests and CI. It is intentionally independent of LangSmith,
Langfuse, or any model provider, so the same test dataset can run without
credentials and then be connected to a hosted evaluation system later.

## Basic Usage

```python
from agentbridge import EvaluationExample, evaluate_agent

report = evaluate_agent(
    agent,
    backend="mock",
    dataset=[
        EvaluationExample(
            input="Check order A123",
            expected_output="eligible",
        ),
    ],
    evaluators={
        "contains_order": lambda example, result: "A123" in str(result.output),
    },
)

assert report.summary["error_count"] == 0
```

An evaluator receives the original `EvaluationExample` and normalized
`RunResult`. It may return a number/boolean, a mapping with `score`, `value`,
`comment`, and `metadata`, or an `EvaluationScore` instance. The resulting
`EvaluationReport` contains one `EvaluationCase` per example, preserving
backend output and per-case failures.

Run the complete example without a provider key:

```bash
python examples/evaluate_refund_agent.py
```

## LangSmith Boundary

LangSmith supports datasets, offline and online evaluations, code evaluators,
LLM-as-judge evaluators, pairwise comparisons, and summary evaluators. The
AgentBridge core contract covers the portable local execution layer first.
The LangChain plugin can attach tracing through its existing observability
configuration, while hosted dataset/evaluator publishing belongs in an
optional integration package so core remains dependency-free.

That integration must preserve the following fields when publishing a run:

- dataset and example identifiers;
- AgentBridge backend and adopted framework version;
- normalized input, output, events, usage, and error;
- evaluator key, score/value, comment, and metadata;
- model route and trace/session identifiers.

This makes a local CI report and a LangSmith experiment comparable without
making hosted LangSmith a required runtime dependency.

## Langfuse Typed Publication And Export

The LangChain plugin publishes reports to Langfuse and exports them through
the [Scores v3 API](https://langfuse.com/docs/api-and-data-platform/features/public-api).
Install the plugin before using these helpers:

```bash
python -m pip install -e plugins/agentbridge-langchain
python examples/langfuse_typed_evaluation.py
```

The example uses an injected transport, not a hosted API. It demonstrates
numeric, boolean, categorical, and text scores, trace subjects, metadata,
paginated export, and matching synchronous/asynchronous results without keys.

For a hosted project, configure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`,
and pass your regional or self-hosted URL as `base_url` when constructing the
client. Then publish an existing report:

```python
from agentbridge_langchain import LangfuseAPIClient, publish_report_scores

client = LangfuseAPIClient(base_url="https://cloud.langfuse.com")
published = publish_report_scores(
    report,
    trace_ids={0: "existing-trace-id"},
    score_types={"verdict": "CATEGORICAL"},
    client=client,
)
scores = list(client.iter_scores_v3(query={"fields": "details,subject"}))
```

`EvaluationScore.value` takes precedence over `score` when present. Booleans,
strings, and numbers infer `BOOLEAN`, `TEXT`, and `NUMERIC`, respectively;
use `score_types` for categorical strings. Comments and score metadata are
preserved, as is example metadata when publishing datasets. Scores without a
value and cases without a mapped trace ID are skipped.

Publishable values are checked for scalar types before any score writes start.
This is not a transaction: network or server validation failures can leave
earlier scores published. The report helper does not assign idempotency IDs;
use the lower-level `create_score()` when controlling retry identity or other
native score targets and options.

Async pipelines use `apublish_report_scores()` and `aiter_scores_v3()`:

```python
from agentbridge_langchain import apublish_report_scores

async def publish_and_export(report, client):
    await apublish_report_scores(
        report, trace_ids={0: "existing-trace-id"}, client=client,
        score_types={"verdict": "CATEGORICAL"},
    )
    return [score async for score in client.aiter_scores_v3(
        query={"fields": "details,subject", "experimentId": "experiment-id"},
    )]
```

Both iterators preserve caller filters, follow opaque server cursors, and raise
an explicit incomplete-export error for cursor cycles or malformed pages.
Records already yielded before an error are partial results, not a complete
export. Request suitable filters to avoid exporting the whole project.
