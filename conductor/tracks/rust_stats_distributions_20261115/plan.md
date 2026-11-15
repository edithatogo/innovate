# Plan: Rust Statistical Distributions and Quantiles Modernization

> **Execution Order:** #5 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_linalg_optimization_20261115`; prerequisite for `rust_nuts_hmc_20261115`).

## Phase 1: Audit, Add `statrs`, and Baseline Characterization Tests
- [x] **1.1 Audit custom distribution/quantile helpers:** Audited `bayesian.rs`, `causal.rs`, and `bootstrap.rs`; recorded `normal_quantile`, `GaussianPriors::log_density`, and `log_likelihood` in `bayesian.rs` (no custom parametric helpers in `causal.rs` or `bootstrap.rs`) in `spec.md`.
- [x] **1.2 Add `statrs` to `rust/crates/innovate-core/Cargo.toml`:** Pinned `statrs = "=0.19.1"` in `rust/crates/innovate-core/Cargo.toml` and verified `cargo check -p innovate-core` succeeds.
- [x] **1.3 Add numerical parity characterization tests:** Added `normal_quantile_matches_statrs_across_tails` and `normal_ln_pdf_matches_unnormalized_gaussian_differences` in `rust/crates/innovate-core/src/bayesian.rs`; verified `cargo test -p innovate-core --locked` for `bayesian` (2 passed), `causal` (10 passed), and `bootstrap` (1 passed).
- [x] **1.4 Conductor Review (Phase 1):** Reviewed `Cargo.toml`, `spec.md`, and `bayesian.rs` characterization tests; committed Phase 1 checkpoint.

## Phase 2: Replace Custom Quantile & Distribution Boilerplate
- [x] **2.1 Replace `bayesian::normal_quantile` and log-PDF helpers:** Updated `rust/crates/innovate-core/src/bayesian.rs` to use `statrs::distribution::{Normal, Continuous, ContinuousCDF}` and preserved `ranked_rhat` and `paired_ess` contracts (fixed constant-chain centering in `paired_ess`). Verified with `cargo test -p innovate-core --locked --test native_boundary bayesian_effective_sample_size`.
- [x] **2.2 Replace `causal.rs` and `bootstrap.rs` distribution helpers:** Confirmed via audit that `causal.rs` and `bootstrap.rs` contain only empirical sample quantiles/critical values and no custom parametric distribution helpers.
- [x] **2.3 Conductor Review (Phase 2):** Reviewed Phase 2 changes and committed checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [x] **3.1 Run full Rust & Python test suites:** Executed `cargo test --workspace --locked` (`innovate-core`: 50 unit + 2 doctests + 104 `native_boundary` + all integration tests, 0 failures) and `uv run --frozen --reinstall-package innovate pytest -o minversion=9.0 tests/native` (366 passed, 5 skipped: 1 live OECD SDMX endpoint, 4 locked optional Arrow extra).
- [x] **3.2 Run docs & security gates:** Executed `cargo fmt --check`, `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, `python3 scripts/check_repository_policy.py`, and `python3 scripts/check_scientific_review_contract.py` (after updating `bayesian.rs` fingerprint) with zero errors/warnings (`cargo-audit` runs in CI).
- [~] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Verified `bayesian.rs` replacement and gates cleanly; Track #5 remains `[~]` in `conductor/tracks.md` (unarchived) pending prerequisite track reconciliation (`rust_http_ureq_20261115`, `rust_tabular_crates_io_20261115`) and coordinator integration in `#624`.
