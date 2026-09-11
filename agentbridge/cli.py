"""Command line interface for AgentBridge."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agentbridge.builtin_tools import build_builtin_tool_registry
from agentbridge.capabilities import capability_matrix
from agentbridge.compare import compare_backends
from agentbridge.manifest import load_agent_spec, load_manifest
from agentbridge.plugins import plugin_status
from agentbridge.registry import adapter_sources, inspect_backend, inspect_backends, list_adapters
from agentbridge.runner import run_agent, stream_agent
from agentbridge.scaffold import scaffold_adapter_plugin
from agentbridge.validation import validate_manifest
from agentbridge.versioning import dependency_versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentbridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-backends", help="List registered backend adapters.")

    inspect_parser = subparsers.add_parser(
        "inspect-backend",
        help="Show capability metadata for one backend or all backends.",
    )
    inspect_parser.add_argument("backend", nargs="?", help="Backend name to inspect.")
    inspect_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    run_parser = subparsers.add_parser("run", help="Run an agent manifest.")
    run_parser.add_argument("--manifest", required=True, help="Path to a JSON/YAML agent manifest.")
    run_parser.add_argument("--backend", required=True, help="Backend adapter name.")
    run_parser.add_argument("--input", required=True, help="Input text for the run.")
    run_parser.add_argument("--stream", action="store_true", help="Stream normalized events.")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    compare_parser = subparsers.add_parser("compare", help="Compare backend fit for a manifest.")
    compare_parser.add_argument("--manifest", required=True, help="Path to a JSON/YAML agent manifest.")
    compare_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend to include. Repeat to compare a subset.",
    )
    compare_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    versions_parser = subparsers.add_parser(
        "versions",
        help="Show adopted and installed framework dependency versions.",
    )
    versions_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest before running.")
    validate_parser.add_argument("--manifest", required=True, help="Path to a JSON/YAML agent manifest.")
    validate_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend to include. Repeat to validate a subset.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Show adapter plugin load status.",
    )
    plugins_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    matrix_parser = subparsers.add_parser(
        "capability-matrix",
        help="Show backend support across canonical AgentBridge capabilities.",
    )
    matrix_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend to include. Repeat to compare a subset.",
    )
    matrix_parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include adapter-reported features not yet in the canonical taxonomy.",
    )
    matrix_format = matrix_parser.add_mutually_exclusive_group()
    matrix_format.add_argument("--json", action="store_true", help="Emit JSON.")
    matrix_format.add_argument("--markdown", action="store_true", help="Emit Markdown.")

    scaffold_parser = subparsers.add_parser(
        "scaffold-plugin",
        help="Create a starter external adapter plugin package.",
    )
    scaffold_parser.add_argument("target_dir", help="Directory where the plugin package is created.")
    scaffold_parser.add_argument("--backend", required=True, help="Backend name, for example google_adk.")
    scaffold_parser.add_argument(
        "--package",
        dest="package_name",
        help="Python package name. Defaults to agentbridge_<backend>.",
    )
    scaffold_parser.add_argument(
        "--distribution",
        dest="distribution_name",
        help="Python distribution name. Defaults to package name with hyphens.",
    )
    scaffold_parser.add_argument("--force", action="store_true", help="Overwrite generated files.")

    args = parser.parse_args(argv)

    try:
        if args.command == "list-backends":
            sources = adapter_sources()
            for backend in list_adapters():
                print(f"{backend}\t{sources.get(backend, 'unknown')}")
            return 0

        if args.command == "inspect-backend":
            payload: Any
            if args.backend:
                payload = inspect_backend(args.backend).model_dump()
            else:
                payload = {
                    backend: capabilities.model_dump()
                    for backend, capabilities in inspect_backends().items()
                }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_capabilities(payload)
            return 0

        if args.command == "run":
            agent = load_agent_spec(args.manifest, tool_registry=build_builtin_tool_registry())
            if args.stream:
                events = [
                    event.model_dump()
                    for event in stream_agent(agent, backend=args.backend, input=args.input)
                ]
                if args.json:
                    print(json.dumps(events, indent=2, sort_keys=True))
                else:
                    for event in events:
                        print(f"{event['type']}: {event['data']}")
            else:
                result = run_agent(agent, backend=args.backend, input=args.input)
                if args.json:
                    print(result.model_dump_json(indent=2))
                else:
                    print(result.output)
            return 0

        if args.command == "compare":
            manifest = load_manifest(args.manifest)
            required_features = _required_features_from_manifest(manifest)
            comparisons = compare_backends(required_features, backends=args.backends)
            payload = {
                "manifest": manifest.name,
                "required_features": required_features,
                "backends": [comparison.as_dict() for comparison in comparisons],
            }
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_comparison(payload)
            return 0

        if args.command == "versions":
            payload = dependency_versions()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_versions(payload)
            return 0

        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            validation = validate_manifest(
                manifest,
                tool_registry=build_builtin_tool_registry(),
                backends=args.backends,
            )
            payload = validation.model_dump()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_validation(payload)
            return 0 if validation.valid else 1

        if args.command == "plugins":
            payload = [result.__dict__ for result in plugin_status()]
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_plugins(payload)
            return 0

        if args.command == "capability-matrix":
            matrix = capability_matrix(
                backends=args.backends,
                include_unknown=args.include_unknown,
            )
            if args.json:
                print(matrix.model_dump_json(indent=2))
            elif args.markdown:
                print(matrix.as_markdown())
            else:
                _print_capability_matrix(matrix.model_dump())
            return 0

        if args.command == "scaffold-plugin":
            created = scaffold_adapter_plugin(
                args.target_dir,
                backend_name=args.backend,
                package_name=args.package_name,
                distribution_name=args.distribution_name,
                force=args.force,
            )
            for item in created:
                print(item.path)
            return 0

    except Exception as exc:
        print(f"agentbridge: error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _print_capabilities(payload: Any) -> None:
    if "backend" in payload:
        _print_one_capability(payload)
        return

    for backend in sorted(payload):
        _print_one_capability(payload[backend])


def _print_one_capability(capability: dict[str, Any]) -> None:
    print(capability["backend"])
    for feature, status in sorted(capability["features"].items()):
        print(f"  {feature}: {status}")
    if capability.get("notes"):
        print("  notes:")
        for feature, note in sorted(capability["notes"].items()):
            print(f"    {feature}: {note}")


def _required_features_from_manifest(manifest: Any) -> list[str]:
    required_features = list(manifest.required_capabilities)
    if manifest.tools and "tools.sync" not in required_features:
        required_features.append("tools.sync")
    if manifest.output_schema and "structured_output" not in required_features:
        required_features.append("structured_output")
    if "agent.instructions" not in required_features:
        required_features.append("agent.instructions")
    return required_features


def _print_comparison(payload: dict[str, Any]) -> None:
    print(f"manifest: {payload['manifest']}")
    print(f"required_features: {', '.join(payload['required_features'])}")
    for backend in payload["backends"]:
        print(f"{backend['backend']}: {backend['fit']} ({backend['score']}/{backend['max_score']})")
        if backend["missing"]:
            print(f"  missing: {', '.join(backend['missing'])}")


def _print_versions(payload: dict[str, dict[str, str | None]]) -> None:
    for name, dependency in sorted(payload.items()):
        installed = dependency["installed_version"] or "not installed"
        print(
            f"{name}: {installed} "
            f"(package: {dependency['package']}, adopted: {dependency['adopted_range']}, "
            f"status: {dependency['status']})"
        )


def _print_validation(payload: dict[str, Any]) -> None:
    status = "valid" if payload["valid"] else "invalid"
    print(f"{payload['manifest']}: {status}")
    print(f"required_features: {', '.join(payload['required_features'])}")
    for backend, fit in sorted(payload["backend_fit"].items()):
        print(f"{backend}: {fit}")
    for message in payload["messages"]:
        print(f"warning: {message}")


def _print_plugins(payload: list[dict[str, Any]]) -> None:
    if not payload:
        print("No adapter plugins discovered.")
        return
    for result in payload:
        status = "loaded" if result["loaded"] else "failed"
        backend = f" -> {result['backend']}" if result.get("backend") else ""
        replaced = " replaced" if result.get("replaced") else ""
        error = f" ({result['error']})" if result.get("error") else ""
        print(f"{result['source']}:{result['name']} {status}{backend}{replaced}{error}")


def _print_capability_matrix(payload: dict[str, Any]) -> None:
    backends = payload["backends"]
    print(f"backends: {', '.join(backends)}")
    current_category = None
    for row in payload["rows"]:
        feature = row["feature"]
        if feature["category"] != current_category:
            current_category = feature["category"]
            print(f"\n{current_category}")
        statuses = ", ".join(
            f"{backend}={row['support'].get(backend, 'unsupported')}" for backend in backends
        )
        print(f"  {feature['key']}: {statuses}")


if __name__ == "__main__":
    raise SystemExit(main())
