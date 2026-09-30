# Upstream Compatibility Pipeline

AgentBridge treats framework compatibility as a continuously maintained contract. A framework
release is not considered supported merely because it installs. The adopted version range, the
exact verified version, the feature inventory, and the adapter tests must move together.

## Scheduled Checks

The weekly `Upstream Compatibility` workflow performs three checks:

- Queries PyPI for the latest release of every adopted framework package.
- Runs core AgentBridge tests and conformance checks on supported Python versions.
- Installs each external adapter in an isolated job and runs its plugin contract tests.
- Follows LangChain's official recursive documentation indexes and detects page additions or removals.

If an upstream release leaves the adopted range, the scheduled check fails and uploads a version
report. That failure is a compatibility-review signal, not proof that the new release is broken.

Run the same version check locally:

```bash
python scripts/check_upstream_versions.py --fail-on-drift
```

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

The current LangChain inventory tracks 1,663 official pages across 13 recursive indexes. Every page
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
