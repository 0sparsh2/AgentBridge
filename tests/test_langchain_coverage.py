from scripts.build_langchain_coverage import classify, validate


def test_classify_assigns_explicit_outcome_to_each_major_area():
    pages = [
        {"url": "https://docs.langchain.com/oss/python/langgraph/persistence.md", "title": "Persistence"},
        {"url": "https://docs.langchain.com/langsmith/evaluation.md", "title": "Evaluation"},
        {"url": "https://docs.langchain.com/langsmith/manage-prompts.md", "title": "Prompts"},
        {"url": "https://docs.langchain.com/oss/typescript/langchain/index.md", "title": "LangChain JS"},
    ]

    decisions = [classify(page) for page in pages]

    assert [decision["status"] for decision in decisions] == [
        "extension",
        "extension",
        "extension",
        "unsupported",
    ]
    assert all(decision["owner"] and decision["action"] for decision in decisions)


def test_validate_rejects_missing_decisions():
    errors = validate({"page_count": 1, "pages": [{"url": "https://example.com"}]})

    assert "https://example.com: missing or invalid status" in errors
