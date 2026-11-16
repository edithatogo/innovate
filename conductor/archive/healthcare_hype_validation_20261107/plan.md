# Hype Impact Implementation Plan

> For agentic workers: execute the named taskbook packets task by task using the required implementation/review workflow.

**Goal:** Fit hype/expectation dynamics and evaluate their impact on innovation, including medicines utilisation.

**Architecture:** Native additive-composite/dynamic models, explicit observation likelihoods, joint inference and uptake-effect coupling.

**Tech Stack:** Rust core, qualified native optimiser/sampler, Polars/Arrow and independent reference tests.

**Spec:** [spec.md](spec.md) and [revised protocol](../../research/healthcare-hype-protocol-20261107.md).

## Global Constraints

No claim-exaggeration annotation prerequisite. No forced universal cycle. No host inference fallback or fabricated posterior. Hype curve estimation is separate from estimating its effect.

## Review Focus

Flattened clipped trough; overlapping unidentifiable pulses; sum extrema differing from pulse centres; shared p/q time-rescaling mistaken for separate mechanisms; reverse causation/common shocks.

## Executable packets

See [taskbook](../../innovation-taskbook-20261107.md) for exact files/signatures/tests/commands.

- [x] H01 — correct composite shape, parameter transforms and derivatives (`models/hype.rs`, `tests/hype_shape.rs`).
- [x] H02 — jointly fit component and total-curve parameters with identifiability diagnostics (`inference/hype_fit.rs`, `tests/hype_fit.rs`).
- [x] H03 — native Bayesian/hierarchical inference and calibration (`inference/hype_posterior.rs`, `tests/hype_posterior.rs`).
- [x] H04 — separate hype effects on adoption, imitation and active use (`models/hype_effects.rs`, `tests/hype_effects.rs`).
- [x] H05 — verify delay semantics and conditional feedback extension (`models/expectation_feedback.rs`, `tests/expectation_feedback.rs`).
- [x] H06 — frozen predictive impact comparison including no-cycle and flexible alternatives (`validation/hype/protocol.json`, `validation/hype/run.py`, `tests/validation/test_hype_protocol.py`).

Source/coverage pilot can proceed now; fit and inference claims require relevant R/I/A/M/V tasks. Venue priority follows innovation/forecasting contribution rather than a mandatory misinformation study.
