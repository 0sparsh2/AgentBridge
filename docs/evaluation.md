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
