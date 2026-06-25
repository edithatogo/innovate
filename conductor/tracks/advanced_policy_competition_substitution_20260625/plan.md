# Advanced Policy, Competition, and Substitution Modeling Plan

## Phase 1: Scientific Gap Audit

- [ ] Task: Inventory advanced modeling capabilities
    - [ ] Audit policy, competition, substitution, network, multi-product, composite, path-dependence, and advanced runtime modules.
    - [ ] Compare APIs against product vision, docs, model cards, capability registry, and tests.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Write failing gap tests
    - [ ] Add tests requiring explicit capability status, docs, model cards, and schema status for each targeted model family.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Scientific Gap Audit' (Protocol in workflow.md)

## Phase 2: Policy and Network Diffusion

- [ ] Task: Implement or promote policy diffusion gaps
    - [ ] Add missing event-history, staggered rollout, spillover, counterfactual, diagnostics, or uncertainty payload behavior identified by the audit.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement or promote network diffusion gaps
    - [ ] Add missing graph traces, intervention diagnostics, optional adapter boundaries, and schema-compatible payloads.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Policy and Network Diffusion' (Protocol in workflow.md)

## Phase 3: Competition and Substitution

- [ ] Task: Implement or promote competition model gaps
    - [ ] Improve multi-product, Lotka-Volterra, market-share attraction, equilibrium, and cross-elasticity behavior where missing.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement or promote substitution model gaps
    - [ ] Improve Fisher-Pry, Norton-Bass, composite substitution, threshold diagnostics, and scenario-comparison payloads where missing.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Competition and Substitution' (Protocol in workflow.md)

## Phase 4: Rust, Polyglot, Docs, and Benchmarks

- [ ] Task: Add schema, Rust, and binding evidence
    - [ ] Add golden fixtures, schema validation, Rust-native or Python-reference status, and binding compatibility evidence for promoted features.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Update docs, tutorials, examples, model cards, and benchmarks
    - [ ] Add Starlight docs, examples, benchmark cases, and model-card metadata for implemented features.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Rust, Polyglot, Docs, and Benchmarks' (Protocol in workflow.md)

## Phase 5: Final Review and Release Gate

- [ ] Task: Run full modeling validation
    - [ ] Run targeted modeling tests, property tests, benchmarks where feasible, and `uv run nox -s lint types tests docs package`.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run conductor-review, push, and monitor CI
    - [ ] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Review and Release Gate' (Protocol in workflow.md)
