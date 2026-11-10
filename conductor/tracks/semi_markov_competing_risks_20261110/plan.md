# Semi-Markov Competing-Risks Diffusion and Cohort Kernels Implementation Plan

**Goal:** Deliver Rust-native competing-risks hazard generators, semi-Markov duration tunnel states, and Polars/Arrow trajectory interchange for multi-product switching and `tobac` dynamic cohort projections.

**Spec:** [spec.md](spec.md).

## Executable packets

- [ ] **SM01** — Continuous-time competing-risks generator $Q$ and uniformization / Padé transition step $P(\Delta t) = \exp(Q\,\Delta t)$ with deterministic semigroup subdivision and row-stochastic conservation property tests.
- [ ] **SM02** — Semi-Markov duration-since-transition tunnel states $\tau \in \{0, \dots, K\}$ and bi-exponential frailty decay kernels reducing to the time-homogeneous Markov baseline when decay parameters vanish.
- [ ] **SM03** — Polars/Arrow zero-copy cohort state-occupancy interchange and consumer parity tests against `edithatogo/tobac` (`src/tobac_supply/dynamic.py` and `src/tobac/health.py` reference oracles).
