# Causal Policy Evaluation Plan

## Phase 1: Causal Contract and Guardrails

- [ ] Task: Define policy evaluation contracts
    - [ ] Add failing tests for intervention timing, comparators, rollout, spillovers, covariates, and estimand metadata.
    - [ ] Implement validated input/output contracts.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add misuse guardrails
    - [ ] Validate missing comparator, leakage, time-window mismatch, and unsupported claim failures.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Causal Contract and Guardrails' (Protocol in workflow.md)

## Phase 2: Evaluation Workflows

- [ ] Task: Implement evaluation summaries
    - [ ] Add pre/post, event-study trajectory, counterfactual, and heterogeneous-effect summaries.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add uncertainty and diagnostics
    - [ ] Add uncertainty metadata, sensitivity notes, and diagnostic warnings.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Evaluation Workflows' (Protocol in workflow.md)

## Phase 3: Docs, Examples, and Claim Policy

- [ ] Task: Add docs and examples
    - [ ] Document causal policy evaluation workflows in Starlight with assumptions and limitations.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Wire model cards and release evidence
    - [ ] Add model-card entries and release-claim safeguards.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Docs, Examples, and Claim Policy' (Protocol in workflow.md)

## Phase 4: Review, Push, and CI

- [ ] Task: Run full causal policy validation
    - [ ] Run targeted tests plus `uv run nox -s lint types tests docs package`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review, push, and monitor CI
    - [ ] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
