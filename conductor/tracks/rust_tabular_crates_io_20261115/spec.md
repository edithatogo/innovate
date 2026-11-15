# Specification: Rust Tabular & Arrow `crates.io` Modernization

## Problem Statement
`rust/crates/innovate-tabular/Cargo.toml` currently pins `polars` and `polars-arrow` to a git revision (`rev = "2d74c435e83b86dc2cf6b71ec6c7b4bd67a4042d"`) with features `["dtype-array", "dtype-struct", "ipc", "ipc_streaming"]`, and the root `Cargo.toml` carries `[patch.crates-io] hashbrown = { git = "https://github.com/pola-rs/hashbrown", ... }`. This blocks `crates.io` publishing, adds heavy compile-time overhead, and pulls in `bincode 2.0.1` (`RUSTSEC-2025-0141`).

## Goals
1. Audit `rust/crates/innovate-tabular/src/lib.rs` and all callers (`innovate-python`, `innovate-ffi`) to determine whether `innovate-tabular` requires Polars query/DataFrame APIs or only Arrow `RecordBatch` / IPC (`ipc`, `ipc_streaming`) interchange.
2. Migrate either to `arrow-ipc` + `arrow-array` + `arrow-schema` (`apache/arrow-rs` from `crates.io`, preferred if only Arrow IPC/schemas are needed) or to a published `crates.io` `polars` release.
3. Remove `[patch.crates-io] hashbrown` from root `Cargo.toml` and eliminate `RUSTSEC-2025-0141`.

## Acceptance Criteria
- Zero `git =` dependencies remain in `rust/crates/innovate-tabular/Cargo.toml` and root `Cargo.toml` `[patch.crates-io]`.
- `cargo test -p innovate-tabular --locked`, `cargo audit` (zero `bincode` advisory), and `pytest tests/native` pass.
