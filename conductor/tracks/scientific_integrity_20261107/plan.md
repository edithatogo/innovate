# Scientific Integrity Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development task by task.

**Goal:** Prevent failed or unsupported calculations from appearing as valid scientific results.

**Architecture:** Share typed inference status and diagnostics across fitters, kernel payloads and reports. Retain independent reference estimators for validation.

**Tech Stack:** Existing Python/JAX/BlackJAX reference paths and the qualified Rust core.

**Spec:** [spec.md](spec.md).

## Global constraints

Follow the [programme](../../frontier-programme-20261107.md); no dependency upgrades or public API removals without a versioned migration. Planning completion is not implementation acceptance.

## Review focus

Failed refits must not retain old posteriors. Empty groups and non-finite values must fail clearly. Unsupported method strings must not select naive estimates. Confounding must not be hidden by result labels. Reports must preserve invalid status.

## Task 1 — Fail-closed uncertainty

Files: `src/innovate/fitters/bayesian_fitter.py`, `src/innovate/fitters/blackjax_fitter.py`, `src/innovate/fitters/diagnostics_contract.py`, `src/innovate/probabilistic.py`; new `tests/unit/test_inference_failure_semantics.py`.

- [ ] Write regressions forcing warmup/sampling failure and a successful-fit-then-failed-refit sequence; assert no posterior samples or credible intervals are returned.
- [ ] Run the focused tests and record the expected failures before changing code.
- [ ] Introduce versioned status semantics and remove synthetic-posterior fallback; make explicitly requested point estimates distinguishable.
- [ ] Verify status propagation through kernel, Arrow and report consumers; commit the tested slice.

## Task 2 — Honest causal dispatch

Files: `src/innovate/causal/policy.py`, `src/innovate/causal/evaluation.py`, `src/innovate/reports/claims.py`; new `tests/unit/test_causal_estimator_dispatch.py`.

- [ ] Test unsupported methods, no overlap, absent groups and a confounded known-truth fixture. Assert an explicit rejection or a verified adjusted result, never silent naive substitution.
- [ ] Run the failing regressions; then implement descriptive/causal separation and supported dispatch.
- [ ] Compare supported estimators with independent reference software and document assumptions; commit.

## Task 3 — Numerical and release qualification

- [ ] Add independent likelihood/gradient references, parameter-recovery cases, simulation-based calibration and interval-coverage experiments with prespecified tolerances.
- [ ] Run `uv run pytest tests/unit/test_inference_failure_semantics.py tests/unit/test_causal_estimator_dispatch.py` and the affected existing fitter/kernel/report suites.
- [ ] Review invalid-result handling across bindings, run required repository gates and record exact commits, commands, seeds and results before archival.
