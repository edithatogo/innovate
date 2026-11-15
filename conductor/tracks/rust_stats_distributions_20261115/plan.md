# Plan: Rust Statistical Distributions and Quantiles Modernization

> **Execution Order:** #5 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_linalg_optimization_20261115`; prerequisite for `rust_nuts_hmc_20261115`).

## Phase 1: Audit, Add `statrs`, and Baseline Characterization Tests
- [ ] **1.1 Audit custom distribution/quantile helpers:** Run `rg -n "normal_quantile|erf|erfc|gamma|ln_gamma|normal_pdf|log_pdf" rust/crates/innovate-core/src/` across `bayesian.rs`, `causal.rs`, and `bootstrap.rs`, and list every function to replace in `spec.md`.
- [ ] **1.2 Add `statrs` to `rust/crates/innovate-core/Cargo.toml`:** Add the latest `crates.io` `statrs` release, run `cargo check -p innovate-core`, and verify `cargo audit` passes.
- [ ] **1.3 Add numerical parity characterization tests:** Add unit tests in `rust/crates/innovate-core/src/bayesian.rs` and `causal.rs` locking tail quantiles (`p = 1e-6, 0.025, 0.5, 0.975, 1-1e-6`), CDFs, and log-PDFs within tight floating-point tolerances. Run `cargo test -p innovate-core --locked bayesian causal bootstrap`.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run tests, and commit checkpoint.

## Phase 2: Replace Custom Quantile & Distribution Boilerplate
- [ ] **2.1 Replace `bayesian::normal_quantile` and log-PDF helpers:** Update `rust/crates/innovate-core/src/bayesian.rs` to use `statrs::distribution::{Normal, Continuous, ContinuousCDF}` while preserving `ranked_rhat` and `paired_ess` contracts. Run `cargo test -p innovate-core --locked bayesian`.
- [ ] **2.2 Replace `causal.rs` and `bootstrap.rs` distribution helpers:** Update `rust/crates/innovate-core/src/causal.rs` and `rust/crates/innovate-core/src/bootstrap.rs` to use `statrs::distribution` (e.g. `Normal`, `StudentsT`). Run `cargo test -p innovate-core --locked causal bootstrap`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native` and confirm zero failures.
- [ ] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
