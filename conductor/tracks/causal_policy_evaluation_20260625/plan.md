# Causal Policy Evaluation Plan

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Causal Contract and Guardrails

- [x] Task: Define policy evaluation contracts (b961c25)
    - [x] Add failing tests for intervention timing, comparators, rollout, spillovers, covariates, and estimand metadata.
    - [x] Implement validated input/output contracts.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add misuse guardrails (b961c25)
    - [x] Validate missing comparator, leakage, time-window mismatch, and unsupported claim failures.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Causal Contract and Guardrails' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Evaluation Workflows

- [x] Task: Implement evaluation summaries
    - [x] Add pre/post, event-study trajectory, counterfactual, and heterogeneous-effect summaries.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add uncertainty and diagnostics
    - [x] Add uncertainty metadata, sensitivity notes, and diagnostic warnings.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Evaluation Workflows' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Docs, Examples, and Claim Policy

- [x] Task: Add docs and examples
    - [x] Document causal policy evaluation workflows in Starlight with assumptions and limitations.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Wire model cards and release evidence
    - [x] Add model-card entries and release-claim safeguards.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Docs, Examples, and Claim Policy' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: pending]

## Phase 4: Review, Push, and CI

- [x] Task: Run full causal policy validation
    - [x] Run targeted tests plus `uv run nox -s lint types tests docs package`.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run conductor-review, push, and monitor CI
    - [x] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
