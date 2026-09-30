import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_langchain_inventory_and_coverage_ledger_are_complete_and_current():
    inventory = json.loads((ROOT / "docs/upstream/langchain-pages.json").read_text())
    coverage = json.loads((ROOT / "docs/upstream/langchain-coverage.json").read_text())
    assert inventory["page_count"] == len(inventory["pages"]) == 1663
    assert coverage["snapshot_page_count"] == coverage["page_count"] == 1663
    assert {page["url"] for page in inventory["pages"]} == {page["url"] for page in coverage["pages"]}
    assert not [page for page in coverage["pages"] if page["status"] == "planned"]
    categories = {page["category"] for page in inventory["pages"]}
    assert {"langchain", "langgraph", "langsmith", "deployment"}.issubset(categories)
