# Specification: Operational Modeling Fixture Contracts

## Overview

Define TreeAge-style decision-tree, state-transition, and discrete-event
simulation fixture contracts for HEOR ecosystem workflows. This track converts
the documented ecosystem follow-ups for TreeAge-style operational modeling and
DES event-log examples into one implementation-ready track.

## Roadmap Source

- `docs/ecosystem/module_incubation_strategy.md`
- `specs/ecosystem/README.md`
- User direction to consider TreeAge and operational modelling such as DES
- XLA strategy guidance that classic DES semantics should not be forced into
  XLA when event dynamics would be distorted

## Functional Requirements

1. Define a TreeAge-style decision-analysis fixture for decision trees and
   state-transition health-economic models used in reimbursement workflows.
2. Define a DES fixture with event logs, queue metrics, resource identifiers,
   pathway states, and simulation run metadata.
3. Represent DES pathways as event logs or simulation run bundles, not private
   engine state.
4. Document XLA eligibility or rejection rationale for operational-modeling
   fixture workflows.
5. Add validation for required fixture fields, schema versions, and
   deterministic fixture sizes.

## Non-Functional Requirements

1. The fixtures must not make `innovate` a health-economic simulation engine.
2. The contracts must remain artifact-first and avoid private engine state.
3. The fixtures must remain small enough for CI and documentation examples.
4. TreeAge-style and DES contracts must be useful to sibling modules without
   adding sibling-project dependencies to the base install.

## Acceptance Criteria

1. Versioned operational-modeling manifests exist under `specs/ecosystem/`.
2. TreeAge-style fixtures include decision-tree or state-transition metadata
   suitable for HTA and reimbursement examples.
3. DES fixtures include event-log and queue-metric payloads with schema and
   provenance metadata.
4. Tests validate required fields, event-log ordering, queue metric columns,
   and XLA rejection or eligibility notes.
5. Ecosystem documentation links the fixture contracts to the adapter promotion
   ladder.

## Out of Scope

1. Implementing a TreeAge parser or proprietary file importer.
2. Implementing a DES engine.
3. Making PM4Py, TreeAge, SimPy, or sibling packages base dependencies.
4. Promoting operational-modeling adapters to supported status.
