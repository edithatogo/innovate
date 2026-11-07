# Advanced Diffusion Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Deliver independently verified advanced modelling capabilities rather than a larger catalogue of labels.

**Architecture:** Compose native dynamics with typed observation models, inference and diagnostics; use Python/R references only as explicit oracles/backends.

**Tech Stack:** Qualified Rust numerical stack, Arrow, benchmark infrastructure and optional specialist inference adapters.

**Spec:** [spec.md](spec.md).

## Global constraints

Relevant integrity, native and benchmark foundations must pass before promotion. Split implementation by model-operation slice; do not make all advanced models one release gate.

## Review focus

Non-identifiable early growth, censored event times, irregular sampling, changing denominators and abandonment/repeated use each require known-truth fixtures.

## Task 1 — Observation and hierarchical inference

Files: `src/innovate/models/contracts.py`, `src/innovate/models/hierarchical.py`, `src/innovate/models/mixture.py`; proposed native model/likelihood modules and `tests/unit/test_observation_semantics.py`.

- [ ] Define count/stock/rate/share contracts and write failing likelihood/constraint tests, including zero counts and incompatible denominators.
- [ ] Implement one likelihood plus hierarchical partial pooling; test shrinkage, parameter recovery and uncertainty against an independent reference.
- [ ] Evaluate held-out groups and early-stage forecasts; commit only the verified slice.

## Task 2 — Dynamics, substitution and networks

Files: `src/innovate/models/advanced.py`, `src/innovate/models/network.py`, `src/innovate/models/policy.py`, `src/innovate/substitute/norton_bass.py`; corresponding proposed native modules.

- [ ] Write recovery and limiting-case tests for each selected extension, including no-network/no-policy, censoring, regime changes and explicit abandonment.
- [ ] Implement one extension at a time, then run independent-oracle, predictive and calibration comparisons from the validation matrix.
- [ ] Reject or retain experimental status when parameters are not recoverable or improvement is not demonstrated.

## Task 3 — Transfer and expensive-inference experiments

- [ ] Register a separate frozen experiment comparing mechanistic/hierarchical baselines with any neural or simulation-based candidate.
- [ ] Test out-of-domain detection, posterior calibration, simulation cost and decision sensitivity; prevent test-set tuning.
- [ ] Review, run native and installed-consumer gates and publish the full comparison before making frontier-performance claims.
