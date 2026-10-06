# Upstream Compatibility Pipeline

AgentBridge treats framework compatibility as a continuously maintained contract. A framework
release is not considered supported merely because it installs. The adopted version range, the
exact verified version, the feature inventory, and the adapter tests must move together.

## Scheduled Checks

The weekly `Upstream Compatibility` workflow performs these checks:

- Queries PyPI for the latest release of every adopted framework package.
- Runs core AgentBridge tests and conformance checks on supported Python versions.
- Installs each external adapter in an isolated job and runs its plugin contract tests.
- Follows LangChain's official recursive documentation indexes and detects page additions or removals.
- Compares the ten adopted Langfuse list/pagination contracts with the published OpenAPI schema.

If an upstream release leaves the adopted range, the scheduled check fails and uploads a version
report. That failure is a compatibility-review signal, not proof that the new release is broken.

Run the same version check locally:

```bash
python scripts/check_upstream_versions.py --fail-on-drift
```

## Langfuse API Contract Drift

Langfuse's cloud API can change independently of a Python SDK package release.
The adopted snapshot in `docs/upstream/langfuse-pagination-contracts.json`
records pagination query types, required query fields, response requirements,
and pagination metadata for observations, scores, experiments, experiment items,
prompts, datasets, dataset items, score configs, annotation queues, and queue
items. It resolves local schema references and
`allOf` inheritance from the official
[OpenAPI schema](https://cloud.langfuse.com/generated/api/openapi.yml).

```bash
python scripts/check_langfuse_api.py --fail-on-drift
python scripts/check_langfuse_api.py --schema /tmp/openapi.yml --fail-on-drift
python scripts/check_langfuse_api.py --snapshot --output /tmp/proposed-langfuse-contracts.json
```

The weekly job uploads a field-level diff. A drift failure means review is
needed; it is not automatically proof of breakage. Fetch/parse failures and
missing endpoints return a separate failed-check status, never a compatibility
success. Description-only changes are ignored. Snapshot mode produces a
proposal and does not overwrite the adopted baseline automatically.

Review API changes against the sync/async pagination tests and
`examples/langfuse_paginated_export.py` before updating the baseline.
This check covers pagination, not every Langfuse endpoint or record field;
hosted execution and broader API conformance remain separate work.

## Release Review Order

When a framework publishes an update:

1. Record the new version and upstream documentation links in `docs/version_policy.md`.
2. Compare the framework's feature and API surface with the AgentBridge capability taxonomy.
3. Add or update the extension configuration needed to expose native options.
4. Add offline contract tests for deterministic behavior.
5. Add credentialed smoke coverage for provider, hosted, deployment, or observability paths when available.

The scheduled workflow also runs isolated CrewAI adapter contracts and credential-free provider,
observability, and comparison-report lanes. Credentialed lanes remain opt-in and are never enabled
by the scheduled job without repository secrets and explicit environment gates.
6. Update the scenario report showing the feature through AgentBridge and the normalized result/events.
7. Update the adopted range only after the exact version passes its adapter checks.

For LangChain-specific review, use [`docs/langchain_coverage.md`](langchain_coverage.md)
and the page snapshot it references. Documentation drift is a compatibility signal:
new pages must be classified and either implemented, exposed as native-only, or
recorded as planned/unsupported.

## Feature Coverage Rule

Every documented framework capability must have one of these explicit outcomes:

- `core`: represented by the shared AgentBridge contract.
- `extension`: available through a framework-specific AgentBridge extension.
- `native_only`: available through the raw backend object with documented escape-hatch access.
- `unsupported`: tracked as a known gap with an issue or roadmap item.

The current LangChain inventory tracks 1,678 official pages across 13 recursive indexes. Every page
has an owner, action, and explicit decision in `docs/upstream/langchain-coverage.json`; no page is
left unassigned. Runtime pages map to adapter contracts or native escape hatches, while hosted
control-plane and product pages remain explicitly external rather than being misrepresented as
portable runtime behavior.

An adapter must not silently discard a documented framework option. The capability matrix,
coverage report, version policy, and scenario reports are the audit trail for that decision.

## Observability and Governance

The same pipeline will eventually run provider-specific checks for LangSmith, Langfuse,
OpenTelemetry, Phoenix, Braintrust, CloudWatch/X-Ray, Google Cloud telemetry, and other native
framework integrations. Those checks remain credentialed and opt-in, while offline tests verify
that callbacks, trace attributes, metadata, run IDs, diagnostics, and raw native objects are
forwarded correctly.
