# Plan: Rust Tabular & Arrow `crates.io` Modernization

> **Execution Order:** #2 (Run in parallel with `rust_http_ureq_20261115` to unblock `crates.io` and clear `RUSTSEC-2025-0141`).

## Phase 1: Usage Audit, Option Selection & Characterization Tests
- [ ] **1.1 Audit `innovate-tabular` API surface:** Read `rust/crates/innovate-tabular/src/lib.rs` and search workspace usages with `rg -n "innovate_tabular|polars" rust/`. Document in `spec.md` whether Polars DataFrame/LazyFrame ops are used or only Arrow IPC / schema / `RecordBatch` serialization.
- [ ] **1.2 Select target `crates.io` dependency:** Choose Option A (`arrow-ipc` + `arrow-array` + `arrow-schema` from `crates.io`) if only Arrow IPC/RecordBatch interchange is used, or Option B (published `crates.io` `polars` release) if Polars DataFrame ops are required. Record the exact version in `spec.md`.
- [ ] **1.3 Add Arrow IPC round-trip characterization tests:** Ensure `rust/crates/innovate-tabular/src/lib.rs` (and `tests/native` Python round-trip tests) cover schema fields, array/struct dtypes, and IPC streaming byte round-trips. Run `cargo test -p innovate-tabular --locked` to establish a passing baseline.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 audit/decision and tests, apply any fixes, and commit checkpoint.

## Phase 2: Replace Git Pins & Remove `hashbrown` Patch
- [ ] **2.1 Update `rust/crates/innovate-tabular/Cargo.toml`:** Replace the `polars` and `polars-arrow` `git = ...` entries with the selected `crates.io` semver dependency (`arrow-*` or published `polars`).
- [ ] **2.2 Remove root `[patch.crates-io]` override:** Delete `[patch.crates-io] hashbrown = ...` from `Cargo.toml` and run `cargo update` so `Cargo.lock` resolves purely from `crates.io` for `innovate-tabular`.
- [ ] **2.3 Refactor `rust/crates/innovate-tabular/src/lib.rs` (if Option A selected):** Update IPC reader/writer and dtype builders to `arrow-ipc` / `arrow-array` / `arrow-schema` while keeping public `innovate-tabular` function signatures and IPC wire schema unchanged.
- [ ] **2.4 Verify `innovate-tabular` and `innovate-python` (Green):** Run `cargo test -p innovate-tabular --locked` and `cargo test -p innovate-python --locked` and confirm all tests pass.
- [ ] **2.5 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run tests, and commit checkpoint.

## Phase 3: Security Audit & Full Workspace Verification
- [ ] **3.1 Verify `RUSTSEC-2025-0141` & `cargo-deny` / scoped `Miri`:** Run `cargo audit` (confirm `bincode 2.0.1` advisory is gone), run `cargo deny check` (confirming `innovate-tabular` has zero `git+` sources), and run `cargo +nightly miri test -p innovate-tabular --lib` on pure-Rust IPC/buffer unit tests to verify zero UB/aliasing errors.
- [ ] **3.2 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native` and confirm zero failures.
- [ ] **3.3 Run docs gate:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-tabular` and confirm zero warnings.
- [ ] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
