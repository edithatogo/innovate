# Ecosystem Module Incubation Plan

## Phase 1: Role And Boundary Definition

- [x] Task: Document current ecosystem roles.
    - [x] Define `innovate` as the adoption and diffusion sibling.
    - [x] Define `lifecourse` as the health-economic simulation consumer or scenario partner.
    - [x] Define `voiage` as the VOI consumer of adoption uncertainty.
    - [x] Define `mars` as a fixed-API optional surrogate/metamodel backend.
    - [x] Define HEOML as the shared artifact extension target for health-economic workflows.
- [x] Task: Document non-goals.
    - [x] Exclude direct sibling-project internals from supported integrations.
    - [x] Exclude pickle from portable interchange.
    - [x] Exclude changes to the `mars` core API.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Role And Boundary Definition' (Protocol in workflow.md)

## Phase 2: Artifact And Extension Contracts

- [ ] Task: Define `innovate` ecosystem artifacts.
    - [ ] Define adoption curves, uptake trajectories, policy-spread traces, network diffusion traces, and diagnostics.
    - [ ] Define parameter uncertainty, calibration, and provenance fields.
    - [ ] Align tabular artifacts with existing Arrow/Parquet interchange.
- [ ] Task: Define HEOML extension alignment.
    - [ ] Map adoption/diffusion artifacts to a future `heoml.extensions.innovate` namespace.
    - [ ] Define when outputs are generic `innovate` artifacts versus HEOML health-economic extension artifacts.
    - [ ] Keep the existing functional-kernel contract as the primary `innovate` execution contract.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Artifact And Extension Contracts' (Protocol in workflow.md)

## Phase 3: Dependency And Promotion Policy

- [ ] Task: Define optional integration gates.
    - [ ] Require optional extras for ecosystem adapters.
    - [ ] Require smoke CI, Renovate coverage, security checks, docs, and removal paths.
    - [ ] Require version compatibility matrices before supported status.
- [ ] Task: Define promotion stages.
    - [ ] Start with documented contract and fixtures.
    - [ ] Promote to experimental adapter only after stable public APIs exist.
    - [ ] Promote to supported adapter only after conformance and release policy are complete.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dependency And Promotion Policy' (Protocol in workflow.md)

## Phase 4: Documentation And Planning Integration

- [ ] Task: Update docs and specs.
    - [ ] Add ecosystem module strategy documentation.
    - [ ] Add the `specs/ecosystem/` contract outline.
    - [ ] Link to the `lifecourse` and `voiage` ecosystem tracks conceptually.
- [ ] Task: Update planning files.
    - [ ] Update `conductor/tracks.md`.
    - [ ] Update `documents/todo.md`.
    - [ ] Update `CHANGELOG.md`.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Documentation And Planning Integration' (Protocol in workflow.md)
