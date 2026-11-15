# Plan: Rust Regime Detection and Changepoint Modernization (`regime.rs`)

> **Execution Order:** #9 (Run after `rust_linalg_optimization_20261115` so per-regime `curve(...)` fitting uses the shared linalg/optimization solver).

## Phase 1: Crate Evaluation & Decision Gate
- [ ] **1.1 Evaluate `augurs-changepoint` and `changepoint`:** Test both `crates.io` crates against `regime::tests::native_grid_matches_independent_ruptures_fixtures` in `rust/crates/innovate-core/src/regime.rs` for exact `jump`-grid and L2 Welford cost parity.
- [ ] **1.2 Record architecture decision in `spec.md`:** Document whether to adopt a changepoint crate (if exact `ruptures` fixture parity holds) or implement exact PELT pruning directly in `regime::detect`.
- [ ] **1.3 Add large-series (`> 2000` observations) benchmark/stability tests:** Add unit tests in `rust/crates/innovate-core/src/regime.rs` locking runtime bounds and numerical stability on long series before refactoring. Run `cargo test -p innovate-core --locked regime`.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 decision and tests, apply any fixes, and commit checkpoint.

## Phase 2: Changepoint Detection & Per-Regime Solver Migration
- [ ] **2.1 Implement chosen changepoint detector in `regime.rs`:** Wire the selected crate or exact PELT pruning into `regime::detect` and verify `cargo test -p innovate-core --locked regime`.
- [ ] **2.2 Migrate per-regime `curve(...)` fitting:** Update `curve(...)` fitting in `rust/crates/innovate-core/src/regime.rs` to call the shared `rust_linalg_optimization_20261115` solver. Run `cargo test -p innovate-core --locked regime`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked regime`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native`.
- [ ] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
