# Model Validation and Benchmark Expansion Plan

## IMPLEMENTATION SUMMARY

This track expanded validation and benchmark capabilities for the innovate library. The implementation adds:

- **Benchmark Corpus**: BenchmarkCase and BenchmarkFamily enums covering DIFFUSION, SUBSTITUTION, and COMPETITION families with reproducible synthetic data
- **Model Cards**: Schema-validated ModelCard dataclass documenting stable model families with assumptions, inputs, outputs, diagnostics, limitations, and benchmark case mappings
- **Benchmark Runner**: BenchmarkRunner harness executing models against benchmark cases with full serialization of runs, metrics, diagnostics, and uncertainty summaries
- **Validation Reporting**: Residual analysis, out-of-sample scoring, sensitivity, uncertainty coverage, and calibration artifacts via DiagnosticsContract integration
- **Leaderboard Artifacts**: Schema-tested benchmark comparison artifacts with reproducibility metadata (dataset version, seed, runtime, dependency versions, backend, hardware, commit)
- **Automation**: Benchmark corpus validation, model-card freshness, and MARS surrogate gate for Rust promotion evidence
- **Documentation**: Starlight benchmark-workflows tutorial covering fast CI checks vs opt-in timing runs, promotion dossier capture, and benchmark interpretation
- **CI Integration**: Tests ensuring all promoted families have benchmark coverage or explicit rationale; Rust benchmarking infrastructure

Key files:
- `src/innovate/benchmarks/` (1255 LOC): corpus, runner, model_cards, automation, mars_surrogate
- `src/innovate/utils/model_validation.py`: Parameter validation for all model families
- `tests/unit/test_benchmark_*.py`: Comprehensive test coverage
- `docs/astro-site/src/content/docs/tutorials/benchmark-workflows.md`: Tutorial documentation
- `docs/source/innovate.benchmarks.*.rst`: API documentation

This track fulfills the 0.5.0 release requirement for scientific validation evidence and model comparison infrastructure.

## Phase 1:

## Phase 1 Checkpoint: [checkpoint: complete] Benchmark Gap Audit

- [x] Task: Inventory benchmark coverage
    - [x] Compare benchmark corpus and model cards against policy, competition, substitution, network, multi-product, and causal surfaces.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Write benchmark coverage tests
    - [x] Add tests requiring benchmark/model-card coverage or explicit rationale for every promoted family.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Benchmark Gap Audit' (Protocol in workflow.md)

## Phase 2 Checkpoint: [checkpoint: complete]

## Phase 2: Validation Artifact Implementation

- [x] Task: Implement validation reports
    - [x] Add residual, out-of-sample, sensitivity, uncertainty, and calibration artifacts.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Implement leaderboard artifacts
    - [x] Add schema-tested benchmark comparison artifacts with reproducibility metadata.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Validation Artifact Implementation' (Protocol in workflow.md)

## Phase 3 Checkpoint: [checkpoint: complete]

## Phase 3: Corpus Expansion and Docs

- [x] Task: Add benchmark cases
    - [x] Add fast metadata benchmark cases for promoted policy, competition, substitution, and network surfaces.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Update Starlight docs and release evidence
    - [x] Document benchmark interpretation and wire evidence into release readiness.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Corpus Expansion and Docs' (Protocol in workflow.md)

## Phase 4 Checkpoint: [checkpoint: complete]

## Phase 4: Review, Push, and CI

- [x] Task: Run benchmark validation
    - [x] Run benchmark metadata tests, targeted model tests, and `uv run nox -s lint types tests docs package`.
    - [x] Commit task changes and attach the required Conductor git note.
- [x] Task: Run conductor-review, push, and monitor CI
    - [x] Apply review fixes, push, and monitor GitHub Actions until green or blocked.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Review, Push, and CI' (Protocol in workflow.md)
