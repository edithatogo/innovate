# Explainability, Sensitivity, and Decision Reports Plan

## Phase 1: Report Contract and Claim Taxonomy

- [ ] Task: Define report schemas and claim taxonomy
    - [ ] Add failing tests for descriptive, predictive, simulation, and causal claim classification.
    - [ ] Implement report envelopes and claim-safety metadata.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add sensitivity input contracts
    - [ ] Define parameter, assumption, timing, and threshold sensitivity inputs.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Report Contract and Claim Taxonomy' (Protocol in workflow.md)

## Phase 2: Sensitivity and Explainability APIs

- [ ] Task: Implement sensitivity analysis helpers
    - [ ] Add parameter perturbation, elasticity, intervention timing, and threshold sensitivity summaries.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement explainability summaries
    - [ ] Add driver, competition, substitution, and policy-component summaries.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Sensitivity and Explainability APIs' (Protocol in workflow.md)

## Phase 3: Reports, Docs, and Evidence

- [ ] Task: Implement JSON and Markdown report export
    - [ ] Add stable exports and examples for policy, competition, and substitution workflows.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run review, push, and CI monitor
    - [ ] Run targeted tests, full nox gates, conductor-review, push, and monitor GitHub Actions.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reports, Docs, and Evidence' (Protocol in workflow.md)
