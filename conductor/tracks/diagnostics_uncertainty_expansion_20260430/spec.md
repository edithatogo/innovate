# Specification: Diagnostics and Uncertainty Expansion

## Overview

Expand diagnostics and uncertainty tooling beyond the completed baseline so model assessment artifacts are durable, schema-compatible, and usable across language bindings. This track turns the roadmap item "richer diagnostics and uncertainty tooling" into a concrete follow-on track.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- Deferred work item: "richer diagnostics and uncertainty tooling"

## Functional Requirements

1. Inventory existing diagnostics, uncertainty summaries, residual checks, and model-comparison outputs.
2. Define a stable artifact contract for residual diagnostics, calibration checks, predictive uncertainty, interval summaries, and model comparison measures.
3. Ensure diagnostics payloads can be serialized through the existing schema and Arrow-compatible interchange layers.
4. Add tests that verify diagnostics artifacts are stable across Python and thin-binding consumers.
5. Document supported diagnostics by model family, backend, and support tier.
6. Define promotion criteria for new diagnostics so experimental tools do not appear stable prematurely.

## Non-Functional Requirements

1. Diagnostics must not depend on private Python object internals when exported through the functional kernel.
2. Outputs must be reproducible or tolerance-bounded for CI validation.
3. Documentation must distinguish implemented diagnostics from planned or experimental diagnostics.
4. Large diagnostic artifacts must avoid excessive memory usage in normal test and package workflows.

## Acceptance Criteria

1. A diagnostics artifact contract is documented and covered by tests.
2. At least one richer diagnostics or uncertainty slice is implemented behind the stable kernel/schema boundary.
3. User documentation describes how to interpret and consume the new artifacts.
4. Cross-language compatibility checks include representative diagnostics payloads.

## Out of Scope

1. Redesigning the full plotting layer.
2. Exposing private estimator internals as public diagnostics.
3. Promoting diagnostics that do not have schema, tests, and documentation.
