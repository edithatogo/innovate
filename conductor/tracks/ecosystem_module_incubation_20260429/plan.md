# Ecosystem Module Incubation Plan

## Phase 1: Role And Boundary Definition [checkpoint: 1a3ae8a]

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
- [x] Task: Conductor - Automated Review and Checkpoint 'Role And Boundary Definition' (Protocol in workflow.md)

## Phase 2: Artifact And Extension Contracts [checkpoint: 5c424fb]

- [x] Task: Define `innovate` ecosystem artifacts.
    - [x] Define adoption curves, uptake trajectories, policy-spread traces, network diffusion traces, and diagnostics.
    - [x] Define parameter uncertainty, calibration, and provenance fields.
    - [x] Align tabular artifacts with existing Arrow/Parquet interchange.
- [x] Task: Define HEOML extension alignment.
    - [x] Map adoption/diffusion artifacts to a future `heoml.extensions.innovate` namespace.
    - [x] Define when outputs are generic `innovate` artifacts versus HEOML health-economic extension artifacts.
    - [x] Keep the existing functional-kernel contract as the primary `innovate` execution contract.
- [x] Task: Conductor - Automated Review and Checkpoint 'Artifact And Extension Contracts' (Protocol in workflow.md)

## Phase 3: Dependency And Promotion Policy [checkpoint: 491489f]

- [x] Task: Define optional integration gates.
    - [x] Require optional extras for ecosystem adapters.
    - [x] Require smoke CI, Renovate coverage, security checks, docs, and removal paths.
    - [x] Require version compatibility matrices before supported status.
- [x] Task: Define promotion stages.
    - [x] Start with documented contract and fixtures.
    - [x] Promote to experimental adapter only after stable public APIs exist.
    - [x] Promote to supported adapter only after conformance and release policy are complete.
- [x] Task: Conductor - Automated Review and Checkpoint 'Dependency And Promotion Policy' (Protocol in workflow.md)

## Phase 4: Documentation And Planning Integration [checkpoint: 69f9956]

- [x] Task: Update docs and specs.
    - [x] Add ecosystem module strategy documentation.
    - [x] Add the `specs/ecosystem/` contract outline.
    - [x] Link to the `lifecourse` and `voiage` ecosystem tracks conceptually.
- [x] Task: Update planning files.
    - [x] Update `conductor/tracks.md`.
    - [x] Update `documents/todo.md`.
    - [x] Update `CHANGELOG.md`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Documentation And Planning Integration' (Protocol in workflow.md)
