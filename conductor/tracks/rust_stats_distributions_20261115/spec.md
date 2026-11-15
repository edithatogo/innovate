# Specification: Rust Statistical Distributions and Quantiles Modernization

## Objective
Replace hand-rolled statistical distribution and quantile helpers (such as `bayesian::normal_quantile` and log-PDF/CDF/normal/Student-t helpers in `bayesian.rs` and `causal.rs`) with `statrs` from `crates.io`.

## Scope
- Replace `normal_quantile` and custom distribution density/CDF calculations with `statrs::distribution` (`Normal`, `StudentsT`, `ChiSquared`, `Beta`, `Gamma`, `Poisson`).
- Preserve existing Vehtari split-R-hat (`ranked_rhat`, `split_rhat_moments`) and bulk-ESS (`paired_ess`) conventions and exact test tolerances.
- Optionally evaluate `nuts-rs` only as a separate follow-on if HMC/NUTS sampling replaces custom MCMC proposals.
