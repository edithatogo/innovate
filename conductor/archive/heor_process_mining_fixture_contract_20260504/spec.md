# Specification: HEOR Process Mining Fixture Contract and Interface Decision

## Overview

The ecosystem process-mining outline reserves PM4Py-style pathway analysis for
a sibling module but lacked a versioned fixture, CLI decision, MCP decision, and
test coverage. This track promotes the outline to a documented-stage portable
contract without adding process mining to the core `innovate` dependency set.

## Functional Requirements

- Add a versioned process-mining fixture bundle under
  `specs/ecosystem/process/fixtures/event_log_v1/`.
- Include deterministic event-log, pathway-discovery, conformance-summary, and
  bottleneck-summary payloads.
- Record PM4Py as a reference candidate only, not a required dependency.
- Decide that CLI support is planned before runtime adapter implementation.
- Decide that MCP remains deferred unless artifacts become agent-queryable or
  workflow-orchestration heavy.
- Add unit tests for fixture shape, ordering, payload consistency, and docs
  links.

## Acceptance Criteria

- The manifest has a stable `schema_version`, `fixture_id`, documented
  promotion stage, dependency policy, CLI surface decision, and MCP deferral.
- Fixture payloads are small, deterministic, and avoid pickle/private Python
  objects.
- Ecosystem docs link the fixture and state the dependency and promotion limits.
- Focused fixture tests pass locally.

## Out of Scope

- Runtime PM4Py adapter implementation.
- Adding PM4Py to core or optional dependencies.
- MCP tool implementation.
- Production CLI commands.
