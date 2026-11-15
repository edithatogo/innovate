# Plan: Rust Statistical Distributions and Quantiles Modernization

> **Execution Order:** #5 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_linalg_optimization_20261115`; prerequisite for `rust_nuts_hmc_20261115`).

## Phase 1: Audit, Add `statrs`, and Baseline Characterization Tests
- [x] **1.1 Audit custom distribution/quantile helpers:** Audited `bayesian.rs`, `causal.rs`, and `bootstrap.rs`; recorded `normal_quantile`, `GaussianPriors::log_density`, and `log_likelihood` in `bayesian.rs` (no custom parametric helpers in `causal.rs` or `bootstrap.rs`) in `spec.md`.
- [x] **1.2 Add `statrs` to `rust/crates/innovate-core/Cargo.toml`:** Pinned `statrs = "=0.19.1"` in `rust/crates/innovate-core/Cargo.toml` and verified `cargo check -p innovate-core` succeeds.
- [x] **1.3 Add numerical parity characterization tests:** Added `normal_quantile_matches_statrs_across_tails` and `normal_ln_pdf_matches_unnormalized_gaussian_differences` in `rust/crates/innovate-core/src/bayesian.rs`; verified `cargo test -p innovate-core --locked` for `bayesian` (2 passed), `causal` (10 passed), and `bootstrap` (1 passed).
- [x] **1.4 Conductor Review (Phase 1):** Reviewed `Cargo.toml`, `spec.md`, and `bayesian.rs` characterization tests; committed Phase 1 checkpoint.

## Phase 2: Replace Custom Quantile & Distribution Boilerplate
- [ ] **2.1 Replace `bayesian::normal_quantile` and log-PDF helpers:** Update `rust/crates/innovate-core/src/bayesian.rs` to use `statrs::distribution::{Normal, Continuous, ContinuousCDF}` while preserving `ranked_rhat` and `paired_ess` contracts. Run `cargo test -p innovate-core --locked bayesian`.
- [ ] **2.2 Replace `causal.rs` and `bootstrap.rs` distribution helpers:** Update `rust/crates/innovate-core/src/causal.rs` and `rust/crates/innovate-core/src/bootstrap.rs` to use `statrs::distribution` (e.g. `Normal`, `StudentsT`). Run `cargo test -p innovate-core --locked causal bootstrap`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native` and confirm zero failures.
- [ ] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
