"""Detect drift in the Langfuse pagination contracts used by AgentBridge."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import yaml


SOURCE = "https://cloud.langfuse.com/generated/api/openapi.yml"
BASELINE = Path(__file__).resolve().parents[1] / "docs/upstream/langfuse-pagination-contracts.json"
ENDPOINTS = (
    "/api/public/v2/observations", "/api/public/v3/scores",
    "/api/public/experiments", "/api/public/experiment-items",
    "/api/public/v2/prompts", "/api/public/v2/datasets", "/api/public/dataset-items",
    "/api/public/score-configs", "/api/public/annotation-queues",
    "/api/public/annotation-queues/{queueId}/items",
)


def resolve_schema(spec: dict, schema: dict, seen: frozenset[str] = frozenset()) -> dict:
    """Resolve local references and inherited metadata without downloading external references."""

    if not isinstance(schema, dict):
        raise ValueError("Invalid OpenAPI schema")
    result = dict(schema)
    parents = list(result.pop("allOf", []))
    reference = result.pop("$ref", None)
    if reference:
        if not isinstance(reference, str) or not reference.startswith("#/") or reference in seen:
            raise ValueError(f"Unsupported or cyclic schema reference: {reference}")
        target = spec
        for part in reference[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        parents.insert(0, resolve_schema(spec, target, seen | {reference}))
    for parent in parents:
        inherited = resolve_schema(spec, parent, seen)
        result["properties"] = {**inherited.get("properties", {}), **result.get("properties", {})}
        result["required"] = sorted(set(inherited.get("required", [])) | set(result.get("required", [])))
        for key, value in inherited.items():
            result.setdefault(key, value)
    return result


def scalar_contract(spec: dict, schema: dict) -> dict:
    resolved = resolve_schema(spec, schema)
    keys = ("type", "nullable", "format", "enum", "minimum", "maximum", "default")
    return {key: resolved[key] for key in keys if key in resolved}


def extract_contracts(spec: dict) -> dict:
    contracts = {}
    for path in ENDPOINTS:
        operation = spec["paths"][path]["get"]
        parameters = [resolve_schema(spec, item) for item in operation.get("parameters", [])]
        queries = [item for item in parameters if item.get("in") == "query"]
        response = resolve_schema(
            spec, operation["responses"]["200"]["content"]["application/json"]["schema"],
        )
        data = resolve_schema(spec, response["properties"]["data"])
        meta = resolve_schema(spec, response["properties"]["meta"])
        contracts[path] = {
            "query_pagination": {
                item["name"]: {"required": item.get("required", False),
                               **scalar_contract(spec, item["schema"])}
                for item in queries if item["name"] in ("cursor", "page", "limit")
            },
            "required_query": sorted(item["name"] for item in queries if item.get("required")),
            "response_required": sorted(response.get("required", [])),
            "data_type": data["type"],
            "metadata_required": sorted(meta.get("required", [])),
            "metadata": {key: scalar_contract(spec, value)
                         for key, value in sorted(meta.get("properties", {}).items())},
        }
    return contracts


def render_report(adopted: dict, current: dict) -> tuple[str, bool]:
    lines = ["# Langfuse API Pagination Compatibility", "",
             f"Source: {SOURCE}", "",
             f"Only the {len(ENDPOINTS)} adopted list/pagination contracts are checked; this is not whole-API conformance.", ""]
    drifted = False
    for path in sorted(set(adopted) | set(current)):
        changed = adopted.get(path) != current.get(path)
        drifted |= changed
        lines.append(f"- `{path}`: {'review needed' if changed else 'unchanged'}")
        if changed:
            before = json.dumps(adopted.get(path), indent=2, sort_keys=True).splitlines()
            after = json.dumps(current.get(path), indent=2, sort_keys=True).splitlines()
            lines.extend(["", "```diff", *difflib.unified_diff(
                before, after, fromfile="adopted", tofile="current", lineterm="",
            ), "```", ""])
    return "\n".join(lines) + "\n", drifted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, help="Replay a local OpenAPI YAML/JSON instead of fetching upstream")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--snapshot", action="store_true", help="Emit a proposed baseline for manual compatibility review")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    try:
        if args.schema:
            spec = yaml.safe_load(args.schema.read_text(encoding="utf-8"))
        else:
            with urlopen(SOURCE, timeout=30) as response:
                spec = yaml.safe_load(response.read())
        current = extract_contracts(spec)
        if args.snapshot:
            report = json.dumps({"source": SOURCE, "contracts": current}, indent=2, sort_keys=True) + "\n"
            drifted = False
        else:
            adopted = json.loads(args.baseline.read_text(encoding="utf-8"))["contracts"]
            report, drifted = render_report(adopted, current)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        report = f"# Langfuse API Pagination Compatibility\n\nCheck failed; compatibility is unverified: {exc}\n"
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        print(report, file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 1 if drifted and args.fail_on_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
