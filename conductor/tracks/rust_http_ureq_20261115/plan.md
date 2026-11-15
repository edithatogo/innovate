# Plan: Rust Synchronous HTTP Client Migration (`ureq`)

> **Execution Order:** #1 (Run first to speed up all subsequent Rust builds). Parallelizable with `rust_tabular_crates_io_20261115`.

## Phase 1: Audit, Lockfile Check, and Characterization Tests
- [ ] **1.1 Verify `reqwest` call sites:** Run `rg -n "reqwest" rust/` and confirm all call sites are confined to `rust/crates/innovate-core/Cargo.toml` and `rust/crates/innovate-core/src/oecd.rs`.
- [ ] **1.2 Add `ureq` dependency and verify lockfile:** Add `ureq` (latest `crates.io` 3.x/2.x release with `rustls` + `ring`, no `native-tls`) to `rust/crates/innovate-core/Cargo.toml` temporarily alongside `reqwest`, run `cargo check -p innovate-core`, and verify `cargo audit` passes with zero new advisories.
- [ ] **1.3 Write characterization tests in `oecd.rs` (Red):** Add unit tests in `rust/crates/innovate-core/src/oecd.rs` covering: (a) non-2xx HTTP status error formatting, (b) redirect rejection (`301`/`302` must not be followed), and (c) request timeout / transport error mapping using a local `std::net::TcpListener` mock server. Run `cargo test -p innovate-core --locked oecd` and confirm tests pass on baseline before migration.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run `cargo test -p innovate-core --locked oecd`, and commit checkpoint.

## Phase 2: Replace `reqwest` with `ureq` in `oecd.rs`
- [ ] **2.1 Swap client implementation in `rust/crates/innovate-core/src/oecd.rs`:** Replace `reqwest::blocking::Client::builder()` and `.redirect(reqwest::redirect::Policy::none())` with `ureq::Agent` configured with `max_redirects(0)` and matching request/global timeout.
- [ ] **2.2 Remove `reqwest` from `rust/crates/innovate-core/Cargo.toml`:** Delete `reqwest = ...` from `[dependencies]` and update `Cargo.lock` via `cargo update -p innovate-core`.
- [ ] **2.3 Verify dependency tree elimination:** Run `cargo tree -p innovate-core` and confirm `reqwest`, `tokio`, `hyper`, `aws-lc-rs`, and `aws-lc-sys` do not appear.
- [ ] **2.4 Run OECD tests (Green):** Execute `cargo test -p innovate-core --locked oecd` and verify all mock-server and parsing tests pass.
- [ ] **2.5 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked oecd`, and commit checkpoint.

## Phase 3: Full Workspace & Gate Verification
- [ ] **3.1 Run full Rust test suite:** Execute `cargo test --workspace --locked` and confirm zero failures.
- [ ] **3.2 Run docs & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace` and `cargo audit` and confirm zero warnings/errors.
- [ ] **3.3 Run Python native tests:** Execute `pytest tests/native` and confirm all tests pass.
- [ ] **3.4 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
