from __future__ import annotations

import json

from scripts import crawl_langchain_docs


def test_crawl_follows_recursive_indexes_and_collects_documentation_pages(monkeypatch):
    documents = {
        "https://docs.langchain.com/llms.txt": """
        - [Build](https://docs.langchain.com/_llms/build.md)
        - [Reference](/oss/python/langchain/reference.md)
        """,
        "https://docs.langchain.com/_llms/build.md": """
        - [LangGraph](https://docs.langchain.com/_llms/langgraph.md)
        - [Agents](https://docs.langchain.com/oss/python/langchain/agents.md)
        """,
        "https://docs.langchain.com/_llms/langgraph.md": """
        - [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api.md)
        """,
    }
    monkeypatch.setattr(crawl_langchain_docs, "fetch", documents.__getitem__)

    snapshot = crawl_langchain_docs.crawl()

    assert snapshot["index_count"] == 3
    assert snapshot["page_count"] == 3
    pages = {page["url"]: page for page in snapshot["pages"]}
    assert pages["https://docs.langchain.com/oss/python/langchain/agents.md"]["category"] == "langchain"
    assert pages["https://docs.langchain.com/oss/python/langgraph/graph-api.md"]["category"] == "langgraph"


def test_compare_reports_added_and_removed_urls(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"pages": [{"url": "https://docs.langchain.com/old.md"}]}),
        encoding="utf-8",
    )
    current = {"pages": [{"url": "https://docs.langchain.com/new.md"}]}

    added, removed = crawl_langchain_docs.compare(current, baseline_path)

    assert added == ["https://docs.langchain.com/new.md"]
    assert removed == ["https://docs.langchain.com/old.md"]
