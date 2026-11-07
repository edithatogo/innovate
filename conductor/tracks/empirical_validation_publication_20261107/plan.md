# Empirical Validation and Publication Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Produce inspectable comparative evidence and a submission-ready software/research package.

**Architecture:** Versioned source manifests and adapters feed frozen benchmark jobs; evaluation artefacts retain all preprocessing, fitting and computational provenance.

**Tech Stack:** Existing benchmark runner, Arrow/Parquet, native model interfaces and independent reference packages.

**Spec:** [spec.md](spec.md).

## Global constraints

No source is accepted without rights, coverage and measurement checks. Freeze analysis choices before confirmation; numerical claims require the integrity gates.

## Review focus

Revision leakage, sales/stock confusion, repeat prescriptions, foundation-model training overlap and correlated forecast errors each need protocol controls and tests.

## Task 1 — Acquire and qualify data

Files: `src/innovate/benchmarks/corpus.py`, `src/innovate/benchmarks/automation.py`, `src/innovate/data/provenance.py`; proposed `benchmarks/manifests/` and `tests/unit/test_benchmark_source_contracts.py`.

- [ ] Audit the matrix sources and existing Hugging Face archives; record exact revisions, licences, units, denominator, temporal grain, missingness, revision policy and exclusions.
- [ ] Write failing tests for duplicate records, revised-vintage leakage, incorrect stock/flow mapping and missing provenance.
- [ ] Implement minimal adapters and immutable manifests; verify fixtures without live-network dependency in fast CI; commit.

## Task 2 — Freeze and execute comparisons

Files: `src/innovate/benchmarks/runner.py`, `src/innovate/benchmarks/model_cards.py`; proposed `benchmarks/protocols/` and `tests/integration/test_external_benchmark_workflow.py`.

- [ ] Prespecify cohorts, horizons, rolling origins, held-out groups, baselines, scores, uncertainty analysis and compute limits.
- [ ] Write leakage and failure-accounting tests; implement reproducible jobs only after recording failing tests.
- [ ] Run pilot checks, then freeze a distinct confirmatory set. Record all methods, failures, exclusions, calibration and costs; retain negative findings.

## Task 3 — Reproduction and publications

Files: existing `paper.qmd`, `references.bib`, `CITATION.cff`; proposed publication-specific manuscript directories and release manifests.

- [ ] Obtain a clean installed-consumer reproduction independent of the development checkout; distinguish agent review from human/external replication.
- [ ] Prepare the scoped JOSS paper and AI disclosure, then separate forecasting/data/decision manuscripts according to the programme.
- [ ] Run focused tests, benchmark reproducibility checks and required repository gates. User/authors handle external submission and editorial correspondence; no submission occurs merely by completing this plan.
