# Full Rust Ownership Completion

## Overview

The current roadmap truth audit correctly avoids claiming full Rust ownership. This track converts that future-state boundary into implementation work by promoting every feasible remaining canonical operation, model family, and stable payload shape into the Rust core, while documenting any intentionally non-native surfaces as durable product boundaries rather than gaps.

## Functional Requirements

1. Promote remaining deterministic model families currently classified as bridge or Python-reference where Rust-native semantics are stable and testable.
2. Implement Rust-native operation coverage for all promoted families, including fit, predict, simulate where deterministic, summarize, diagnostics, and serialization round trips.
3. Promote stable payload shapes that do not require Python object semantics, including selected covariate matrices, event split summaries, fitted-state summaries, and deterministic simulation policies.
4. Keep non-stable surfaces explicitly outside the Rust core when they depend on posterior objects, arbitrary Python callbacks, graph/agent mutable state, or experimental stochastic policies.
5. Update Python, Rust, and binding-layer dispatch so promoted operations use the Rust implementation consistently.
6. Refresh machine-readable ownership artifacts and claim-language tests so release documentation may only claim what the evidence supports.

## Non-Functional Requirements

1. Rust APIs must remain ABI-conscious and additive unless a major-version migration is explicitly approved.
2. Python ergonomics must not regress; Python remains the primary research interface.
3. Performance-sensitive promoted paths must include benchmark evidence and regression thresholds.
4. Error payloads and schema validation must be language-neutral and stable.
5. Implementation must preserve deterministic parity with the Python reference within documented numerical tolerances.

## Acceptance Criteria

1. `docs/source/_static/rust_full_ownership_validation.json` either allows a full Rust ownership claim or lists only intentional non-core boundaries with owner, rationale, and revisit criteria.
2. Rust test coverage exists for every promoted operation and payload shape.
3. Python tests prove dispatch parity and fallback behavior.
4. Binding smoke tests prove no promoted Rust operation is unavailable from language clients.
5. Benchmarks show no unacceptable regression against the previous promoted Rust slices.
6. The final track review confirms there is no stale wording that implies unverified full ownership.

## Required Operational Cadence

Every implementation task in this track must be completed with:

1. A task implementation commit.
2. A separate plan-status commit that records the task commit SHA.
3. A phase-end `conductor-review` run, review-fix commit if needed, push, and GitHub Actions monitoring until checks pass.
4. A final whole-track `conductor-review`, final push, and GitHub Actions pass before archive.

## Out of Scope

1. Replacing Python as the primary user interface.
2. Promoting unstable posterior, graph, agent, or callback-heavy surfaces without a separate design decision.
3. Breaking existing public package names or registry identities.
