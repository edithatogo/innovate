# Plan: Rust Adaptive ODE/IVP Integration Modernization

> **Execution Order:** #6 (Run after `rust_linalg_optimization_20261115` to avoid concurrent edits to `numerics.rs`).

## Phase 1: Crate Evaluation & Characterization Tests
- [x] **1.1 Locate all `rk4` / ODE call sites:** Run `rg -n "rk4" rust/crates/innovate-core/src/` across `numerics.rs`, `competition.rs`, `substitution.rs`, and `mechanistic.rs`.
- [x] **1.2 Evaluate `ode_solvers` vs `differential-equations`:** Benchmark both `crates.io` crates on irregular-grid and stiff coupled diffusion systems, verify `cargo audit` and `cargo doc` compatibility, record the choice in `spec.md`, and add it to `rust/crates/innovate-core/Cargo.toml`.
- [x] **1.3 Add characterization & metamorphic tests:** Add tests in `rust/crates/innovate-core/src/competition.rs`, `substitution.rs`, and `mechanistic.rs` comparing baseline outputs on regular grids and asserting scale/step-invariance on irregular grids and extreme rate constants. Run `cargo test -p innovate-core --locked competition substitution mechanistic`.
- [x] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run tests, and commit checkpoint.

## Phase 2: Adaptive ODE Solver Migration
- [x] **2.1 Implement adaptive ODE wrapper in `rust/crates/innovate-core/src/numerics.rs`:** Add a deterministic dense-output IVP helper using the selected crate (`Dormand-Prince` / `BDF` fallback) returning exact evaluations at caller-requested time points.
- [x] **2.2 Migrate `competition.rs`, `substitution.rs`, and `mechanistic.rs`:** Replace `numerics::rk4` calls with the new adaptive solver helper and remove dead fixed-step `rk4` code if unused. Run `cargo test -p innovate-core --locked competition substitution mechanistic`.
- [x] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [x] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native`.
- [x] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit`.
- [x] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
