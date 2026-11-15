# Specification: Rust Statistical Distributions and Quantiles Modernization

## Objective
Replace hand-rolled statistical distribution and quantile helpers (such as `bayesian::normal_quantile` and log-PDF/CDF/normal/Student-t helpers in `bayesian.rs` and `causal.rs`) with `statrs` from `crates.io`.

## Scope
- Pin `statrs = "=0.19.1"` in `rust/crates/innovate-core/Cargo.toml`.
- Audit findings across `bayesian.rs`, `causal.rs`, and `bootstrap.rs`:
  - `rust/crates/innovate-core/src/bayesian.rs`: `normal_quantile` (48-step bisection over `libm::erfc`), `GaussianPriors::log_density`, and `log_likelihood` (custom `-0.5 * ((x - mu) / sd).powi(2)` log-densities). Replace with `statrs::distribution::{Normal, Continuous, ContinuousCDF}` (`inverse_cdf` and `ln_pdf`).
  - `rust/crates/innovate-core/src/causal.rs` and `rust/crates/innovate-core/src/bootstrap.rs`: no custom parametric distribution/quantile helpers (`erf`/`erfc`/`normal_quantile`/`StudentsT`); both only compute empirical sample quantiles/critical values.
- Preserve existing Vehtari split-R-hat (`ranked_rhat`, `split_rhat_moments`) and bulk-ESS (`paired_ess`) conventions and exact test tolerances.
- Optionally evaluate `nuts-rs` only as a separate follow-on if HMC/NUTS sampling replaces custom MCMC proposals.
