# Plan: Rust Linear Algebra and Nonlinear Optimization Modernization

> **Execution Order:** #4 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_stats_distributions_20261115`; prerequisite for `rust_ode_integration_20261115` and `rust_regime_changepoint_20261115`).

## Phase 1: Crate Selection, Lockfile Verification & Characterization Tests
- [ ] **1.1 Locate all linear algebra & optimization call sites:** Run `rg -n "fn solve|fn local_rank|damped|Vec<Vec<f64>>" rust/crates/innovate-core/src/` and list all call sites in `numerics.rs`, `causal.rs`, `fitting.rs`, and `regime.rs`.
- [ ] **1.2 Evaluate and pin `crates.io` stack:** Compare `nalgebra` + `levenberg-marquardt` vs `faer` + `argmin` for `rust-version = "1.99.1"`, `RUSTSEC` cleanliness (`cargo audit`), and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps`. Record the chosen crates and exact versions in `spec.md` and add them to `rust/crates/innovate-core/Cargo.toml`.
- [ ] **1.3 Add characterization tests (Red/Baseline):** Add unit tests in `rust/crates/innovate-core/src/numerics.rs` locking current outputs and diagnostics for `numerics::solve`, `numerics::local_rank`, and multistart nonlinear least-squares on well-conditioned, ill-conditioned, and rank-deficient fixtures. Run `cargo test -p innovate-core --locked numerics` to verify baseline pass.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run `cargo test -p innovate-core --locked numerics`, and commit checkpoint.

## Phase 2: Migrate `numerics.rs` & `causal.rs`
- [ ] **2.1 Replace `numerics::solve` and `numerics::local_rank`:** Refactor `rust/crates/innovate-core/src/numerics.rs` to use contiguous matrix decompositions (pivoted QR / SVD) from the chosen linalg crate and eliminate intermediate `Vec<Vec<f64>>` allocations. Run `cargo test -p innovate-core --locked numerics`.
- [ ] **2.2 Replace custom damped least-squares optimizer:** Migrate the multistart least-squares solver in `rust/crates/innovate-core/src/numerics.rs` (and callers in `fitting.rs`) to `levenberg-marquardt` / `argmin`, preserving deterministic multistart tie-breaking and diagnostic fields. Run `cargo test -p innovate-core --locked fitting`.
- [ ] **2.3 Migrate `causal.rs` matrix operations:** Update regression/covariance matrix operations in `rust/crates/innovate-core/src/causal.rs` to use the shared linalg crate. Run `cargo test -p innovate-core --locked causal`.
- [ ] **2.4 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust test suite:** Execute `cargo test --workspace --locked` and confirm zero failures.
- [ ] **3.2 Run Python native tests:** Execute `pytest tests/native` and confirm all native tests pass.
- [ ] **3.3 Run docs, `Miri` & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, scoped `cargo +nightly miri test -p innovate-core --lib numerics` (pure-Rust kernel checks), `cargo audit`, and `cargo deny check` (if configured), confirming zero warnings/advisories.
- [ ] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
