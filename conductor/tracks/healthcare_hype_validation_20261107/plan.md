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

- [ ] H01 — correct composite shape, parameter transforms and derivatives.
- [ ] H02 — jointly fit component and total-curve parameters with identifiability diagnostics.
- [ ] H03 — native Bayesian/hierarchical inference and calibration.
- [ ] H04 — separate hype effects on adoption, imitation and active use.
- [ ] H05 — verify delay semantics and conditional feedback extension.
- [ ] H06 — frozen predictive impact comparison including no-cycle and flexible alternatives.

Source/coverage pilot can proceed now; fit and inference claims require relevant R/I/A/M/V tasks. Venue priority follows innovation/forecasting contribution rather than a mandatory misinformation study.
