# Plan: Rust HMC/NUTS Posterior Sampling Migration (`nuts-rs`)

- [ ] Evaluate `nuts-rs` version compatibility with Rust 1.99.1, `Cargo.lock`, `cargo audit`, and `docs.rs` builds.
- [ ] Design unconstrained parameterizations (log/logit bijections + Jacobian log-det adjustments) and gradient evaluation (analytic or central finite differences) for Bass, Logistic, Gompertz, and Norton–Bass posteriors.
- [ ] Implement `CpuLogpFunc` + `CpuMath` multi-chain seeded sampler wrapper in `bayesian.rs` and wire chain outputs into `diagnostic_chains`, `ranked_rhat`, and `paired_ess`.
- [ ] Add posterior recovery, determinism (same seed -> identical draws), and divergence-reporting tests in Rust and `tests/native`.
- [ ] Verify full Rust (`cargo test --locked`), Python (`pytest tests/native`), and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc` before review and archive.
