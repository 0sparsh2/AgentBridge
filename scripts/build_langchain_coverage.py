"""Build and validate a page-level LangChain coverage ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SNAPSHOT = Path("docs/upstream/langchain-pages.json")
DEFAULT_OUTPUT = Path("docs/upstream/langchain-coverage.json")


def classify(page: dict[str, str]) -> dict[str, str]:
    url = page["url"]
    title = page.get("title", "")
    lower = f"{url} {title}".lower()

    if "/oss/typescript/" in url or "/oss/javascript/" in url:
        return {
            "status": "unsupported",
            "owner": "python-sdk",
            "action": "Keep outside the Python AgentBridge SDK scope; track a separate TypeScript SDK if needed.",
        }
    if "/oss/python/deepagents/" in url or url.endswith("/oss/python/reference/deepagents-python.md"):
        if any(token in lower for token in ("ag-ui", "a2a", "acp", "/frontend/")):
            return {
                "status": "native_only",
                "owner": "deepagents-adapter",
                "action": "Preserve the native Deep Agents protocol/runtime object; add a protocol-specific AgentBridge adapter when its contract is selected.",
            }
        return {
            "status": "extension",
            "owner": "deepagents-adapter",
            "action": "Use the optional Deep Agents adapter and native_options pass-through; add a focused contract fixture when a page introduces a new runtime option.",
        }
    if "/oss/deepagents/code/" in url or "managed-deep-agents" in url:
        return {
            "status": "planned",
            "owner": "runtime-integrations",
            "action": "Keep separate from the local Deep Agents adapter; define hosted/code-runtime credentials and lifecycle contracts first.",
        }
    if "/_llms/" in url or url.endswith("openapi.json"):
        return {
            "status": "native_only",
            "owner": "docs-index",
            "action": "Use as upstream navigation/schema input; do not expose index files as runtime features.",
        }
    if any(token in lower for token in ("trace", "tracing", "observability", "log-", "feedback")):
        return {
            "status": "extension",
            "owner": "observability",
            "action": "Use AgentBridge metadata/events plus the optional provider integration; add a focused adapter test when the page describes a new option.",
        }
    if any(token in lower for token in ("evaluation", "evaluator", "dataset", "experiment")):
        return {
            "status": "extension",
            "owner": "evaluation",
            "action": "Use EvaluationExample/EvaluationReport locally or the optional LangSmith publisher; add provider-specific fields when required.",
        }
    if "prompt" in lower:
        return {
            "status": "extension",
            "owner": "prompt-management",
            "action": "Use the optional LangSmith prompt pull/push bridge and preserve the native prompt object for LangChain formatting.",
        }
    if any(token in lower for token in ("playground", "studio")):
        return {
            "status": "planned",
            "owner": "prompt-management",
            "action": "Add prompt references, commit/tag resolution, and a native client escape hatch without putting hosted credentials in core.",
        }
    if any(token in lower for token in ("deepagent", "deep-agent", "fleet", "gateway", "managed")):
        return {
            "status": "planned",
            "owner": "runtime-integrations",
            "action": "Create an optional integration only after the hosted/runtime boundary and credential contract are documented.",
        }
    if any(token in lower for token in ("deploy", "deployment", "agent-server", "self-host", "control-plane", "data-plane")):
        return {
            "status": "planned",
            "owner": "deployment",
            "action": "Expose deployment metadata and native SDK/API clients; do not make a hosted control plane a core dependency.",
        }
    if "/oss/python/langgraph/" in url or "/oss/python/langchain/" in url:
        return {
            "status": "extension",
            "owner": "langchain-adapter",
            "action": "Map the documented Python runtime option to AgentBridge config, native pass-through, or a normalized event/result with a contract test.",
        }
    return {
        "status": "planned",
        "owner": "ecosystem-review",
        "action": "Review the page and assign it to a concrete adapter, integration, escape hatch, or scope decision.",
    }


def build(snapshot_path: Path = SNAPSHOT) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pages = []
    for page in snapshot["pages"]:
        pages.append({**page, **classify(page)})
    return {
        "source": snapshot["source"],
        "snapshot_page_count": snapshot["page_count"],
        "page_count": len(pages),
        "pages": pages,
    }


def validate(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pages = ledger.get("pages", [])
    if ledger.get("page_count") != len(pages):
        errors.append("page_count does not match pages length")
    seen: set[str] = set()
    allowed = {"core", "extension", "native_only", "planned", "unsupported"}
    for page in pages:
        url = page.get("url")
        if not url:
            errors.append("page without url")
        elif url in seen:
            errors.append(f"duplicate page: {url}")
        seen.add(url)
        if page.get("status") not in allowed:
            errors.append(f"{url}: missing or invalid status")
        if not page.get("owner") or not page.get("action"):
            errors.append(f"{url}: missing owner/action")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    ledger = build(args.snapshot)
    errors = validate(ledger)
    if errors:
        for error in errors:
            print(error)
        return 1
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != ledger:
            print(f"Coverage ledger is stale: regenerate {args.output}")
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(f"Validated coverage decisions for {ledger['page_count']} LangChain pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
