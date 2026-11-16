# Specification: Rust Synchronous HTTP Client Migration (`ureq`)

## Problem Statement
`rust/crates/innovate-core/Cargo.toml` currently depends on `reqwest = { version = "=0.13.5", default-features = false, features = ["blocking", "rustls"] }` solely for synchronous blocking HTTP calls in `rust/crates/innovate-core/src/oecd.rs`. Even with `blocking`, `reqwest` pulls in `tokio`, `hyper`, `aws-lc-rs`, and `aws-lc-sys` (which requires `cmake` and compiles C/assembly), inflating cold compile times and CI cache footprints.

## Goals
1. Replace `reqwest` with `ureq` (published `crates.io` release, pure-Rust `rustls` with `ring` provider) in `rust/crates/innovate-core/Cargo.toml`.
2. Preserve exact `oecd.rs` behavior: explicit timeouts, zero redirects (`Policy::none()` equivalent via `max_redirects(0)`), status/body error mapping, and deterministic parsing.
3. Verify `tokio`, `hyper`, `aws-lc-rs`, and `aws-lc-sys` are no longer in `cargo tree -p innovate-core`.

## Acceptance Criteria
- Zero `reqwest` references remain in `rust/`.
- `cargo test -p innovate-core --locked`, `cargo audit`, and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps -p innovate-core` pass.
