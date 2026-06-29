# Data Ingestion and Provenance Connectors Plan

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: pending] Dataset Contract Design

- [x] Task: Define dataset schemas
    - [x] Add failing tests for adoption, substitution, competition, policy timing, and network-edge datasets.
    - [x] Implement validated contracts and provenance metadata.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Add validation diagnostics
    - [x] Implement missingness, monotonicity, time alignment, denominator, duplicate, and unit checks.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dataset Contract Design' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: pending]

## Phase 2: Ingestion Helpers and Connectors

- [x] Task: Implement local ingestion helpers
    - [x] Support CSV, Parquet/Arrow, and Polars inputs with reproducible artifacts.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement public-data-style adapter
    - [x] Add a documented adapter pattern with provenance and licensing safeguards.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Ingestion Helpers and Connectors' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: pending]

## Phase 3: Docs, Benchmarks, and Release Evidence

- [x] Task: Integrate with benchmarks and scenarios
    - [x] Link validated datasets to benchmark cases, model cards, and scenario workflows.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run review, push, and CI monitor
    - [x] Run targeted tests, full nox gates, conductor-review, push, and monitor GitHub Actions.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Docs, Benchmarks, and Release Evidence' (Protocol in workflow.md)
