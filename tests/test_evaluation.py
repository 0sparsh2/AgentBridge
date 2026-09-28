from agentbridge import AgentSpec, EvaluationExample, evaluate_agent


def test_evaluate_agent_normalizes_scores_and_summary():
    agent = AgentSpec(
        name="echo",
        instructions="Echo the request.",
        model="agentbridge/offline",
    )

    report = evaluate_agent(
        agent,
        backend="mock",
        dataset=[EvaluationExample(input="one"), EvaluationExample(input="two")],
        evaluators={"contains_output": lambda example, result: example.input == result.output["input"]},
        dataset_name="echo-regression",
    )

    assert report.dataset_name == "echo-regression"
    assert report.summary == {
        "example_count": 2,
        "success_count": 2,
        "error_count": 0,
        "score_means": {"contains_output": 1.0},
    }
    assert [case.scores[0].score for case in report.cases] == [1.0, 1.0]


def test_evaluate_agent_preserves_case_failures():
    agent = AgentSpec(
        name="echo",
        instructions="Echo the request.",
        model="agentbridge/offline",
    )

    report = evaluate_agent(
        agent,
        backend="missing-backend",
        dataset=[EvaluationExample(input="one")],
    )

    assert report.summary["error_count"] == 1
    assert report.cases[0].error.startswith("AdapterNotFoundError:")
