# Plan: Rust Tabular & Arrow `crates.io` Modernization

> **Execution Order:** #2 (Run in parallel with `rust_http_ureq_20261115` to unblock `crates.io` and clear `RUSTSEC-2025-0141`).

## Phase 1: Usage Audit, Option Selection & Characterization Tests
- [x] **1.1 Audit `innovate-tabular` API surface:** Audited `rust/crates/innovate-tabular/src/lib.rs`; public functions accept/return `polars_core::prelude::DataFrame` and `PolarsResult` alongside Arrow IPC byte exchange.
- [x] **1.2 Select target `crates.io` dependency:** Selected Option B (`polars-core = "=0.55.2"`, `polars-arrow = "=0.55.2"` from `crates.io`); recorded in `spec.md`.
- [x] **1.3 Add Arrow IPC round-trip characterization tests:** `tests/prediction.rs` (11 tests) and `tests/scenario_exports.rs` (8 tests) cover schema fields, nested JSON/struct/list types, and IPC file/stream round-trips.
- [x] **1.4 Conductor Review (Phase 1):** Verified `cargo test -p innovate-tabular --locked` passes.

## Phase 2: Replace Git Pins & Remove `hashbrown` Patch
- [x] **2.1 Update `rust/crates/innovate-tabular/Cargo.toml`:** `polars-core = "=0.55.2"` and `polars-arrow = "=0.55.2"` pinned from `crates.io` with zero `git =` entries.
- [x] **2.2 Remove root `[patch.crates-io]` override:** Root `Cargo.toml` has no `[patch.crates-io]` section and `Cargo.lock` has zero `bincode` entries.
- [x] **2.3 Refactor `rust/crates/innovate-tabular/src/lib.rs` (if Option A selected):** N/A (Option B selected; `DataFrame` & Arrow IPC contracts preserved).
- [x] **2.4 Verify `innovate-tabular` and `innovate-python` (Green):** `cargo test -p innovate-tabular --locked` (19 integration tests) and `cargo test -p innovate-python --locked` passed.
- [x] **2.5 Conductor Review (Phase 2):** Verified clean manifest and test passes.

## Phase 3: Security Audit & Full Workspace Verification
- [x] **3.1 Verify `RUSTSEC-2025-0141` & `cargo-deny` / scoped `Miri`:** `bincode` is absent from `Cargo.lock`, and `cargo deny check advisories licenses bans` passed (`advisories ok, bans ok, licenses ok`).
- [x] **3.2 Run full Rust & Python test suites:** `cargo test --workspace --locked` and `.venv/bin/pytest tests/native` (`371 passed, 1 skipped`) passed.
- [x] **3.3 Run docs gate:** `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` passed with zero warnings.
- [x] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Reconciled and marked complete in `conductor/tracks.md`.
