# Specification: Rust Linear Algebra and Nonlinear Optimization Modernization

## Objective
Replace hand-rolled `Vec<Vec<f64>>` dense linear algebra, local rank deficiency checks, and custom damped least-squares optimization in `rust/crates/innovate-core/src/numerics.rs` and `rust/crates/innovate-core/src/causal.rs` with audited, published `crates.io` crates (`nalgebra` or `faer`, plus `levenberg-marquardt` or `argmin`).

## Scope
- Selected `crates.io` stack: `nalgebra = "=0.34.2"` (`Apache-2.0`, `rust-version = 1.87.0`) and `levenberg-marquardt = "=0.15.0"` (`MIT`, built on `nalgebra 0.34` so `Cargo.lock` resolves a single unified `nalgebra` version).
- Replace `numerics::solve` (custom Gaussian elimination) and `numerics::local_rank` (custom rank/column-norm checks) with contiguous matrix decompositions (pivoted QR / SVD / LU).
- Replace custom finite-difference `jacobian` and multistart damped least-squares loop with `levenberg-marquardt` or `argmin` while preserving deterministic multistart semantics and diagnostic JSON outputs.
- Migrate `causal.rs` matrix solves/covariance operations off nested `Vec<Vec<f64>>`.
- Require published `crates.io` dependency versions (no `git` dependencies) and preserve `cargo test --locked -p innovate-core -p innovate-tabular -p innovate-ffi` and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc`.

## Non-Goals
- Changing public JSON/FFI/Python/R/Julia schemas or relaxing numerical tolerance contracts.
- Modifying `regime.rs` changepoint detection (`detect`) where `ruptures` parity is required.
