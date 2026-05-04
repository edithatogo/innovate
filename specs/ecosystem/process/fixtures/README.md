# HEOR Process Mining Fixtures

This directory is reserved for small deterministic fixtures for HEOR process
mining. Any future fixture should be reproducible, portable, and small enough
for CI.

Planned fixture families:

- event-log samples
- pathway discovery outputs
- conformance summaries
- bottleneck analyses

Versioned fixture:

- [event_log_v1/manifest.json](./event_log_v1/manifest.json) defines the first
  deterministic documented-stage process-mining bundle. It covers event-log
  rows, pathway discovery output, conformance summaries, bottleneck summaries,
  CLI expectations, an explicit MCP deferral, and PM4Py as a reference
  candidate rather than a required dependency.

The fixture contract is documented only. Runtime process-mining adapters remain
future work until optional extras, smoke CI, security checks, documentation,
and compatibility matrices are in place.
