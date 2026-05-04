# Implementation Plan: Operational Modeling Fixture Contracts

## Phase 1: TreeAge-Style Fixture Contract

- [x] Task: Define decision-analysis fixture fields
    - [x] Specify decision-tree, state-transition, strategy, state, transition, and payoff metadata
    - [x] Define provenance and schema-version fields
    - [x] State that proprietary TreeAge parsing is out of scope
- [x] Task: Add TreeAge-style fixture tests
    - [x] Write failing tests for required decision-model fields and schema metadata
    - [x] Check that the fixture is artifact-first and does not require proprietary tools
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: TreeAge-Style Fixture Contract' (Protocol in workflow.md)

## Phase 2: DES Fixture Contract

- [x] Task: Define DES event-log and queue-metric fields
    - [x] Specify event traces, pathway states, resource identifiers, queue times, and run metadata
    - [x] Require deterministic ordering rules and provenance metadata
    - [x] Document why dynamic event semantics should not be forced into XLA when unsuitable
- [x] Task: Add DES fixture tests
    - [x] Write failing tests for event-log columns, queue metrics, ordering, and schema version
    - [x] Check XLA eligibility or rejection notes are present
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: DES Fixture Contract' (Protocol in workflow.md)

## Phase 3: Documentation And Promotion Gates

- [x] Task: Update ecosystem documentation
    - [x] Link TreeAge-style and DES fixtures from ecosystem docs
    - [x] Document optional dependency and adapter promotion gates
    - [x] Keep runtime simulation engines out of the current `innovate` package scope
- [x] Task: Run validation gates
    - [x] Run focused operational-model fixture tests
    - [x] Run relevant docs or prose checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation And Promotion Gates' (Protocol in workflow.md)
