# Semi-Markov Competing-Risks Diffusion and Cohort Kernels Implementation Plan

**Goal:** Deliver Rust-native competing-risks hazard generators, semi-Markov duration tunnel states, and Polars/Arrow trajectory interchange for multi-product switching and `tobac` dynamic cohort projections.

**Spec:** [spec.md](spec.md).

## Executable packets

- [x] **SM01** — Continuous-time competing-risks generator $Q$ and uniformization / Padé transition step $P(\Delta t) = \exp(Q\,\Delta t)$ with deterministic semigroup subdivision and row-stochastic conservation property tests (`models/semi_markov.rs`, `tests/semi_markov_kernels.rs`).
- [x] **SM02** — Semi-Markov duration-since-transition tunnel states $\tau \in \{0, \dots, K\}$ and bi-exponential frailty decay kernels reducing to the time-homogeneous Markov baseline when decay parameters vanish (`models/semi_markov.rs`, `tests/semi_markov_kernels.rs`).
- [~] **SM03** — Polars/Arrow cohort state-occupancy interchange (`cohort_occupancy_ipc`/`decode_cohort_occupancy_ipc` in `innovate-tabular`, `tests/cohort_interchange.rs` complete; downstream `edithatogo/tobac` consumer parity remains in external `tobac` repo).
