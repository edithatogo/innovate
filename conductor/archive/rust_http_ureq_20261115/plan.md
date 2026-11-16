# Plan: Rust Synchronous HTTP Client Migration (`ureq`)

> **Execution Order:** #1 (Run first to speed up all subsequent Rust builds). Parallelizable with `rust_tabular_crates_io_20261115`.

## Phase 1: Audit, Lockfile Check, and Characterization Tests
- [x] **1.1 Verify `reqwest` call sites:** Ran `rg -n "reqwest" rust/` and `rg -n "reqwest" Cargo.lock rust/`; zero `reqwest` references remain.
- [x] **1.2 Add `ureq` dependency and verify lockfile:** `rust/crates/innovate-core/Cargo.toml` pins `ureq = { version = "=3.5.2", default-features = false, features = ["rustls"] }`; `cargo deny check advisories licenses bans` passed.
- [x] **1.3 Write characterization tests in `oecd.rs` (Red):** `rust/crates/innovate-core/src/oecd.rs` covers non-2xx HTTP status (`429`), redirect rejection (`301`/`302`), oversized/invalid-UTF8 responses, and timeout/transport error mapping (`maps_timeout_and_transport_errors`).
- [x] **1.4 Conductor Review (Phase 1):** Verified baseline and characterization tests via `cargo test -p innovate-core --locked oecd`.

## Phase 2: Replace `reqwest` with `ureq` in `oecd.rs`
- [x] **2.1 Swap client implementation in `rust/crates/innovate-core/src/oecd.rs`:** `download_with_timeout` uses `ureq::Agent` with `max_redirects(0)`, `http_status_as_error(false)`, and 30s global timeout.
- [x] **2.2 Remove `reqwest` from `rust/crates/innovate-core/Cargo.toml`:** `reqwest` removed from `Cargo.toml` and `Cargo.lock`.
- [x] **2.3 Verify dependency tree elimination:** `cargo tree -p innovate-core` confirms `ureq v3.5.2` with zero `reqwest`/`tokio`/`hyper`/`aws-lc-rs`/`aws-lc-sys`.
- [x] **2.4 Run OECD tests (Green):** `cargo test -p innovate-core --locked oecd` passed (3 unit tests, 2 integration tests, 1 ignored live endpoint test).
- [x] **2.5 Conductor Review (Phase 2):** Verified clean diff and OECD behavior.

## Phase 3: Full Workspace & Gate Verification
- [x] **3.1 Run full Rust test suite:** `cargo test --workspace --locked` passed with zero failures.
- [x] **3.2 Run docs & security gates:** `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo deny check advisories licenses bans` passed (`advisories ok, bans ok, licenses ok`).
- [x] **3.3 Run Python native tests:** `.venv/bin/pytest tests/native` passed (`371 passed, 1 skipped` for live OECD endpoint).
- [x] **3.4 Conductor Review (Phase 3 & Track Sign-off):** All local gates verified; marked complete in `conductor/tracks.md`.
