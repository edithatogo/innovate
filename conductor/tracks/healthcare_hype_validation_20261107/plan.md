# Healthcare Hype Evaluation Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Establish a falsifiable measurement and forecasting study without presupposing a hype cycle.

**Architecture:** Versioned attention/claim/evidence/utilisation observations feed explicit statistical comparators; mechanistic extensions remain gated by identifiability and held-out performance.

**Tech Stack:** Existing source/provenance interfaces, Arrow, independent statistical baselines and qualified native model slices.

**Spec:** [spec.md](spec.md); [study protocol](../../research/healthcare-hype-protocol-20261107.md).

## Global constraints

Public aggregate data first. No diagnosis of hype from search volume or attention–uptake disparity alone. No causal language from lagged correlation. No implied product/indication mapping absent source support.

## Review focus

Imposed cycle shape, positive-only time rescaling, inconsistent delayed forcing, media-source leakage and repeat prescriptions each require explicit tests or study controls.

## Task 1 — Source and construct feasibility

Files: proposed `research/healthcare_hype/protocol.md`, `research/healthcare_hype/source_manifest.json`, `research/healthcare_hype/annotation_codebook.md`; existing `src/innovate/data/provenance.py`.

- [ ] Audit intervention/indication/geography definitions and coverage before selecting the final cohort; include low-attention and unsuccessful candidates where ascertainable.
- [ ] Pilot the claim-to-evidence annotation rubric with independent blinded reviewers and record disagreements, adjudication and uncertainty.
- [ ] Freeze rights-permitted source identifiers, date semantics, duplicate/syndication handling, revision policy and a separate holdout. Register protocol amendments before confirmatory analysis.

## Task 2 — Verify and constrain existing hype models

Files: `src/innovate/hype/hype_cycle.py`, `src/innovate/hype/hype_modified_bass.py`, `src/innovate/hype/delayed_hype_bass.py`, `src/innovate/abm/sentiment_hype_cycle.py`; proposed `tests/unit/test_hype_model_semantics.py`.

- [ ] Write failing tests for invalid widths/time grids, zero/no-hype limits, negative delayed visibility, numerical stability, shared-p/q time-rescaling and inconsistent delay semantics.
- [ ] Record failures, then add validation and explicit model limitations. Replace the delay solver only after demonstrating mathematical/numerical equivalence for the intended contract.
- [ ] Implement any inhibition, separate p/q effects, multi-peak dynamics or abandonment as new identifiable/versioned mechanisms, not unannounced changes to old parameters; verify against independent references.

## Task 3 — Measurement and predictive evaluation

Files: proposed `research/healthcare_hype/evaluate.py`, `tests/integration/test_healthcare_hype_workflow.py`; existing benchmark/report contracts.

- [ ] Compare naive, no-attention diffusion, distributed-lag/GAM or state-space, restricted hype-cycle and optional mechanistic models under frozen temporal/technology holdouts.
- [ ] Test synthetic no-peak, one-peak, multi-peak, inhibitory, capacity-limited and common-cause cases. Estimate uncertainty in annotations and model outputs.
- [ ] Report incremental predictive value, calibration, failures and robustness. Reserve causal analysis for a separate identified design.
- [ ] Run focused/repository gates and prepare the measurement study for JMIR Infodemiology or a broader validated contribution for JMIR. Do not mark empirical validity or native ownership complete from this plan.
