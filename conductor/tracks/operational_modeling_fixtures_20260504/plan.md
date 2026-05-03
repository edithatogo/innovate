# Implementation Plan: Operational Modeling Fixture Contracts

## Phase 1: TreeAge-Style Fixture Contract

- [ ] Task: Define decision-analysis fixture fields
    - [ ] Specify decision-tree, state-transition, strategy, state, transition, and payoff metadata
    - [ ] Define provenance and schema-version fields
    - [ ] State that proprietary TreeAge parsing is out of scope
- [ ] Task: Add TreeAge-style fixture tests
    - [ ] Write failing tests for required decision-model fields and schema metadata
    - [ ] Check that the fixture is artifact-first and does not require proprietary tools
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: TreeAge-Style Fixture Contract' (Protocol in workflow.md)

## Phase 2: DES Fixture Contract

- [ ] Task: Define DES event-log and queue-metric fields
    - [ ] Specify event traces, pathway states, resource identifiers, queue times, and run metadata
    - [ ] Require deterministic ordering rules and provenance metadata
    - [ ] Document why dynamic event semantics should not be forced into XLA when unsuitable
- [ ] Task: Add DES fixture tests
    - [ ] Write failing tests for event-log columns, queue metrics, ordering, and schema version
    - [ ] Check XLA eligibility or rejection notes are present
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: DES Fixture Contract' (Protocol in workflow.md)

## Phase 3: Documentation And Promotion Gates

- [ ] Task: Update ecosystem documentation
    - [ ] Link TreeAge-style and DES fixtures from ecosystem docs
    - [ ] Document optional dependency and adapter promotion gates
    - [ ] Keep runtime simulation engines out of the current `innovate` package scope
- [ ] Task: Run validation gates
    - [ ] Run focused operational-model fixture tests
    - [ ] Run relevant docs or prose checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation And Promotion Gates' (Protocol in workflow.md)
