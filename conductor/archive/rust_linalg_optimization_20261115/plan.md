# Plan: Rust Linear Algebra and Nonlinear Optimization Modernization

> **Execution Order:** #4 (Start after `rust_http_ureq_20261115` and `rust_tabular_crates_io_20261115`; parallelizable with `rust_stats_distributions_20261115`; prerequisite for `rust_ode_integration_20261115` and `rust_regime_changepoint_20261115`).

## Phase 1: Crate Selection, Lockfile Verification & Characterization Tests
- [x] **1.1 Locate all linear algebra & optimization call sites:** Run `rg -n "fn solve|fn local_rank|damped|Vec<Vec<f64>>" rust/crates/innovate-core/src/` and list all call sites in `numerics.rs`, `causal.rs`, `fitting.rs`, and `regime.rs`.
- [x] **1.2 Evaluate and pin `crates.io` stack:** Compare `nalgebra` + `levenberg-marquardt` vs `faer` + `argmin` for `rust-version = "1.99.1"`, `RUSTSEC` cleanliness (`cargo audit`), and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps`. Record the chosen crates and exact versions in `spec.md` and add them to `rust/crates/innovate-core/Cargo.toml`.
- [x] **1.3 Add characterization tests (Red/Baseline):** Add unit tests in `rust/crates/innovate-core/src/numerics.rs` locking current outputs and diagnostics for `numerics::solve`, `numerics::local_rank`, and multistart nonlinear least-squares on well-conditioned, ill-conditioned, and rank-deficient fixtures. Run `cargo test -p innovate-core --locked numerics` to verify baseline pass.
- [x] **1.4 Conductor Review (Phase 1):** Verified `cargo check -p innovate-core`, `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-core`, and `cargo test -p innovate-core --locked numerics` (11 passed).

## Phase 2: Migrate `numerics.rs` & `causal.rs`
- [x] **2.1 Replace `numerics::solve` and `numerics::local_rank`:** Refactored `rust/crates/innovate-core/src/numerics.rs` to use `nalgebra::LU` (`#[cfg(test)] solve`) and `nalgebra::ColPivQR` (`local_rank`) on contiguous `DMatrix<f64>`.
- [x] **2.2 Replace custom damped least-squares optimizer:** Migrated `numerics::least_squares` to `levenberg_marquardt::LevenbergMarquardt` via `LeastSquaresProblemAdapter`, preserving deterministic multistart tie-breaking and diagnostic fields.
- [x] **2.3 Migrate `causal.rs` matrix operations:** Migrated `fit_logistic_propensities` in `rust/crates/innovate-core/src/causal.rs` to contiguous `nalgebra::DMatrix` / `DVector`.
- [x] **2.4 Conductor Review (Phase 2):** Verified `cargo test -p innovate-core --locked causal && cargo test -p innovate-core --locked` (0 warnings, all passed).

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [x] **3.1 Run full Rust test suite:** Executed `cargo test --workspace --locked` (`innovate-core`: 51 unit + 2 doctests, `innovate-tabular`: 11 `prediction` + 8 `scenario_exports`, `innovate-ffi` + `innovate-python` workspace checks) with zero failures.
- [x] **3.2 Run Python native tests:** Executed `uv run --frozen --reinstall-package innovate pytest -o minversion=9.0 tests/native` — 366 passed, 5 skipped (1 live OECD SDMX endpoint test, 4 locked optional Arrow extra tests), 0 failed.
- [x] **3.3 Run docs, `Miri` & security gates:** Executed `cargo fmt --check`, `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, and `python3 scripts/check_repository_policy.py` with zero errors/warnings (`no deny.toml`; `miri` and `cargo-audit` are not installed locally and run in CI).
- [x] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Reviewed diff against `1cd9af80`, applied `.with_stepbound(3.0)` + `with_patience(max_iterations)` + `max_iterations == 1` termination fixes in `rust/crates/innovate-core/src/numerics.rs`, re-ran verification gates cleanly, and marked track complete in `conductor/tracks.md`.
