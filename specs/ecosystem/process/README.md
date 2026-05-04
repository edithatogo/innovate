# HEOR Process Mining Outline

This directory reserves the ecosystem contract outline for health economics and
outcomes research process-mining use cases. It is the home for PM4Py-style
pathway analysis in the ecosystem, but it is not a commitment to add process
mining into the current `innovate` package.

## Scope

- event-log ingestion for HEOR pathways
- pathway discovery and conformance checking
- bottleneck and variant analysis
- conformance metrics for care-flow and implementation pathways
- portable event-log and trace summaries for sibling modules

## Non-Goals

- replacing the current `innovate` diffusion/adoption API
- adding process mining as a core dependency
- tying the outline to private Python objects

## Contract Notes

- CLI support is planned before runtime adapter implementation. Initial command
  concepts are `validate-event-log`, `summarize-pathway`, and
  `export-conformance-summary`.
- MCP is deferred unless the module becomes agent-queryable or
  workflow-orchestration heavy.
- PM4Py is the current reference candidate, but not a required dependency.

## Versioned Fixtures

- [fixtures/event_log_v1/manifest.json](./fixtures/event_log_v1/manifest.json)
  defines a documented-stage portable event-log bundle with pathway discovery,
  conformance, and bottleneck summary payloads. The bundle is synthetic, small,
  deterministic, and independent of PM4Py, pickle, and private Python objects.
