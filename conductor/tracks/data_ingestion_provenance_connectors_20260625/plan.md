# Data Ingestion and Provenance Connectors Plan

## Phase 1: Dataset Contract Design

## Phase 1 Checkpoint: [checkpoint: pending] Dataset Contract Design

- [~] Task: Define dataset schemas
    - [ ] Add failing tests for adoption, substitution, competition, policy timing, and network-edge datasets.
    - [ ] Implement validated contracts and provenance metadata.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Add validation diagnostics
    - [ ] Implement missingness, monotonicity, time alignment, denominator, duplicate, and unit checks.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Dataset Contract Design' (Protocol in workflow.md)

## Phase 2: Ingestion Helpers and Connectors

## Phase 2 Checkpoint: [checkpoint: pending] Ingestion Helpers and Connectors

- [ ] Task: Implement local ingestion helpers
    - [ ] Support CSV, Parquet/Arrow, and Polars inputs with reproducible artifacts.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Implement public-data-style adapter
    - [ ] Add a documented adapter pattern with provenance and licensing safeguards.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Ingestion Helpers and Connectors' (Protocol in workflow.md)

## Phase 3: Docs, Benchmarks, and Release Evidence

## Phase 3 Checkpoint: [checkpoint: pending] Docs, Benchmarks, and Release Evidence

- [ ] Task: Integrate with benchmarks and scenarios
    - [ ] Link validated datasets to benchmark cases, model cards, and scenario workflows.
    - [ ] Commit task changes and attach the required Conductor git note.
- [ ] Task: Run review, push, and CI monitor
    - [ ] Run targeted tests, full nox gates, conductor-review, push, and monitor GitHub Actions.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Docs, Benchmarks, and Release Evidence' (Protocol in workflow.md)
