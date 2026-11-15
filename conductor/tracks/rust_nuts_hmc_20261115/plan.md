# Plan: Rust HMC/NUTS Posterior Sampling Migration (`nuts-rs`)

> **Execution Order:** #8 (Run after `rust_stats_distributions_20261115` so `bayesian.rs` priors/diagnostics are already migrated onto `statrs`).

## Phase 1: `nuts-rs` Compatibility Check & Parameterization Design
- [x] **1.1 Verify `nuts-rs` compatibility:** Add `nuts-rs` (`crates.io` release) to `rust/crates/innovate-core/Cargo.toml`, run `cargo check -p innovate-core`, `cargo audit`, and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-core`, and record the compatible version in `spec.md`.
- [x] **1.2 Design unconstrained bijections & gradients:** In `spec.md`, specify log/logit transforms + Jacobian log-determinant adjustments and gradient evaluation (analytic or central finite differences) for Bass, Logistic, Gompertz, and Norton–Bass posteriors in `rust/crates/innovate-core/src/bayesian.rs`.
- [x] **1.3 Add baseline posterior & determinism characterization tests:** Add unit tests in `rust/crates/innovate-core/src/bayesian.rs` checking synthetic parameter recovery, seed determinism (same seed -> identical draws), and `ranked_rhat` / `paired_ess` output shapes. Run `cargo test -p innovate-core --locked bayesian`.
- [x] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 design and tests, apply any fixes, and commit checkpoint.

## Phase 2: Implement `CpuLogpFunc` + `CpuMath` Sampler in `bayesian.rs`
- [x] **2.1 Implement `CpuLogpFunc` for diffusion models:** Implement unconstrained `logp` and gradient evaluation with Jacobian adjustments in `rust/crates/innovate-core/src/bayesian.rs`.
- [x] **2.2 Implement multi-chain seeded `nuts-rs` wrapper:** Wire `CpuMath` multi-chain seeded sampling into `bayesian.rs` and map chain draws, divergences, and step sizes into `diagnostic_chains`, `ranked_rhat`, and `paired_ess`.
- [x] **2.3 Verify Rust posterior & determinism tests (Green):** Run `cargo test -p innovate-core --locked bayesian` and verify posterior recovery, identical draws for identical seeds, and divergence reporting.
- [x] **2.4 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked bayesian`, and commit checkpoint.

## Phase 3: Polyglot Integration & Gate Verification
- [x] **3.1 Add Python native tests for NUTS diagnostics:** Add/update tests in `tests/native` verifying `diagnostic_chains`, `ranked_rhat`, `paired_ess`, and divergence fields through Python bindings. Run `pytest tests/native`.
- [x] **3.2 Run full Rust, docs & security gates:** Execute `cargo test --workspace --locked`, `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, and `cargo audit`.
- [x] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
