---
title: Advanced runtime workflows
description: Ensemble, policy, streaming, calibration, and accelerator-aware advanced runtime workflows.
---

Advanced runtime workflows are opt-in surfaces for production analytics and
forecast evaluation. They return stable `AdvancedResult` payloads where the
contract is promoted, and explicit experimental metadata where evidence is still
being gathered.

## Stable surfaces

- `policy_scenario` compares baseline and intervention trajectories with
  auditable assumptions, covariates, incremental effect, and final lift.
- `uncertainty_calibration` returns calibrated intervals, residual diagnostics,
  overall coverage, and holdout coverage.

## Experimental surfaces

- `regime_ensemble` combines compatible adoption trajectories with weights and
  score diagnostics.
- `streaming_update` appends new cumulative observations while preserving
  incremental state metadata.

## Runnable example

Run `examples/advanced_runtime_workflows.py` to build one report covering every
advanced runtime workflow. The example does not require JAX, Rust-native
bindings, or probabilistic extras.

## Accelerator evidence

The performance smoke record lives in
`docs/source/_static/advanced_runtime/performance_evidence.json`. It records
NumPy execution as the dependency-free path and documents safe fallback when
optional JAX or Rust-native paths are unavailable.
