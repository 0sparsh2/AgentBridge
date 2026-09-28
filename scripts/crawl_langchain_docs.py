"""Inventory the official LangChain documentation indexes.

LangChain publishes a recursive ``/_llms/`` documentation index. This script
follows those indexes without scraping the rendered site, records every linked
documentation page, and can detect upstream additions/removals in CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

DEFAULT_ROOT = "https://docs.langchain.com/llms.txt"
DEFAULT_SNAPSHOT = Path("docs/upstream/langchain-pages.json")
DOCS_HOST = "docs.langchain.com"
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "AgentBridge-doc-inventory/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def canonical_url(base: str, raw: str) -> str | None:
    url, _fragment = urldefrag(urljoin(base, raw.strip()))
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != DOCS_HOST:
        return None
    return url


def is_index(url: str) -> bool:
    return url == DEFAULT_ROOT or "/_llms/" in url and url.endswith(".md")


def category_for(url: str) -> str:
    path = urlparse(url).path
    if "/langgraph/" in path:
        return "langgraph"
    if "/langchain/" in path:
        return "langchain"
    if "/deepagents/" in path or "/deep-agents/" in path:
        return "deep_agents"
    if "/langsmith/" in path or "lang-smith" in path:
        return "langsmith"
    if "/fleet/" in path:
        return "fleet"
    if "/gateway/" in path or "llm-gateway" in path:
        return "llm_gateway"
    if "/deploy" in path or "/deployment" in path or "agent-server" in path:
        return "deployment"
    if "/oss/" in path:
        return "open_source_other"
    if "/_llms/" in path:
        return "index"
    return "platform_other"


def crawl(root: str = DEFAULT_ROOT) -> dict[str, object]:
    queue: deque[str] = deque([root])
    visited_indexes: set[str] = set()
    pages: dict[str, dict[str, str]] = {}

    while queue:
        index_url = queue.popleft()
        if index_url in visited_indexes:
            continue
        visited_indexes.add(index_url)
        text = fetch(index_url)
        for title, raw_url in LINK_RE.findall(text):
            url = canonical_url(index_url, raw_url)
            if url is None:
                continue
            if is_index(url):
                if url not in visited_indexes:
                    queue.append(url)
                continue
            path = urlparse(url).path
            if not (path.endswith(".md") or path.endswith(".json")):
                continue
            pages[url] = {
                "title": " ".join(title.split()),
                "category": category_for(url),
                "source_index": index_url,
            }

    return {
        "source": root,
        "index_count": len(visited_indexes),
        "page_count": len(pages),
        "indexes": sorted(visited_indexes),
        "pages": [
            {"url": url, **pages[url]}
            for url in sorted(pages)
        ],
    }


def write_snapshot(snapshot: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def compare(current: dict[str, object], baseline_path: Path) -> tuple[list[str], list[str]]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_urls = {page["url"] for page in current["pages"]}
    baseline_urls = {page["url"] for page in baseline["pages"]}
    return sorted(current_urls - baseline_urls), sorted(baseline_urls - current_urls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--check", action="store_true", help="compare against --baseline")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    snapshot = crawl(args.root)
    added: list[str] = []
    removed: list[str] = []
    if args.check:
        if not args.baseline.exists():
            print(f"Baseline does not exist: {args.baseline}", file=sys.stderr)
            return 2
        added, removed = compare(snapshot, args.baseline)

    if args.output:
        write_snapshot(snapshot, args.output)
    else:
        print(json.dumps(snapshot, indent=2))

    print(
        f"LangChain docs: {snapshot['page_count']} pages across "
        f"{snapshot['index_count']} recursive indexes.",
        file=sys.stderr,
    )
    if added or removed:
        print(f"Added: {len(added)}; removed: {len(removed)}", file=sys.stderr)
        for url in added[:20]:
            print(f"+ {url}", file=sys.stderr)
        for url in removed[:20]:
            print(f"- {url}", file=sys.stderr)
    if args.fail_on_drift and (added or removed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
