# Advanced Policy, Competition, and Substitution Modeling Plan

## Phase 1: Scientific Gap Audit

- [x] Task: Inventory advanced modeling capabilities
    - [x] Audit policy, competition, substitution, network, multi-product, composite, path-dependence, and advanced runtime modules.
    - [x] Compare APIs against product vision, docs, model cards, capability registry, and tests.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write failing gap tests (ae14a10)
    - [x] Add tests requiring explicit capability status, docs, model cards, and schema status for each targeted model family.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Scientific Gap Audit' (Protocol in workflow.md)
    - [x] Gap coverage tests written and pass (9/10)
    - [x] Starlight docs deferred to Phase 4

## Phase 2: Policy and Network Diffusion

- [x] Task: Implement or promote policy diffusion gaps (38df8be)
    - [x] Added set_intervention_nodes() to NetworkDiffusionModel
    - [x] Added network intervention diagnostics
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement or promote network diffusion gaps (38df8be)
    - [x] Added graph traces infrastructure
    - [x] Added intervention node API
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Policy and Network Diffusion' (Protocol in workflow.md)

## Phase 3: Competition and Substitution

- [x] Task: Implement or promote competition model gaps (38df8be)
    - [x] Added equilibrium() to LotkaVolterraCompetition and MultiProductDiffusionModel
    - [x] Added cross_elasticity() to competition module
    - [x] Added LockInModel to capability registry
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement or promote substitution model gaps (38df8be)
    - [x] Added threshold_diagnostics() to FisherPryModel and NortonBassModel
    - [x] Added replacement threshold analysis
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Competition and Substitution' (Protocol in workflow.md)

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
