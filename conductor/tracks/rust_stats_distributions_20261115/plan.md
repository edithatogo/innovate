# Plan: Rust Statistical Distributions and Quantiles Modernization

> **Execution Order:** #5 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_linalg_optimization_20261115`; prerequisite for `rust_nuts_hmc_20261115`).

## Phase 1: Audit, Add `statrs`, and Baseline Characterization Tests
- [x] **1.1 Audit custom distribution/quantile helpers:** Completed against main `42189a223fd44c823ca11aec689ea67b11c25790`; actual inventory, retained score/percentile semantics and private probability-domain proof are recorded in `spec.md`.
- [!] **1.2 Add `statrs` to `rust/crates/innovate-core/Cargo.toml`:** Sole integrator owns the manifest, lock and shared dependency qualification. Exact request and feature boundary are recorded in [ownership.md](ownership.md). No lane source result closes this prerequisite.
- [~] **1.3 Add numerical parity characterization tests:** Independent fixture and Bayesian/bootstrap baseline tests are implemented. Dependency CDF/log-PDF qualification follows; causal changes remain integrator-owned. Run one valid Cargo filter per command (the original multi-positional-filter command is not valid Cargo syntax).
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run tests, and commit checkpoint.

## Phase 2: Replace Custom Quantile & Distribution Boilerplate
- [ ] **2.1 Replace `bayesian::normal_quantile` and log-PDF helpers:** Update `rust/crates/innovate-core/src/bayesian.rs` to use `statrs::distribution::{Normal, Continuous, ContinuousCDF}` while preserving `ranked_rhat` and `paired_ess` contracts. Run `cargo test -p innovate-core --locked bayesian`.
- [ ] **2.2 Replace `causal.rs` and `bootstrap.rs` distribution helpers:** Update `rust/crates/innovate-core/src/causal.rs` and `rust/crates/innovate-core/src/bootstrap.rs` to use `statrs::distribution` (e.g. `Normal`, `StudentsT`). Run `cargo test -p innovate-core --locked causal bootstrap`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native` and confirm zero failures.
- [ ] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
