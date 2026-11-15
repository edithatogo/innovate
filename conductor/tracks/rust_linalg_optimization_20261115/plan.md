# Plan: Rust Linear Algebra and Nonlinear Optimization Modernization

- [ ] Benchmark and select published `crates.io` stack (`nalgebra` + `levenberg-marquardt` vs `faer` + `argmin`), verifying `Cargo.lock`, `RUSTSEC`, and `docs.rs` compatibility.
- [ ] Add characterization tests locking current `numerics::solve`, `numerics::local_rank`, and multistart convergence diagnostics on ill-conditioned and rank-deficient fixtures.
- [ ] Replace `numerics::solve` and `numerics::local_rank` with contiguous matrix decompositions (pivoted QR / SVD) and remove `Vec<Vec<f64>>` allocations.
- [ ] Replace custom damped least-squares optimizer with `levenberg-marquardt` / `argmin`, preserving deterministic multistart tie-breaking and diagnostic fields.
- [ ] Migrate `causal.rs` regression/covariance matrix operations to the selected linear algebra crate.
- [ ] Run full Rust (`cargo test --locked`), Python (`pytest tests/native`), `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc`, and `cargo audit` checks before review and archive.
