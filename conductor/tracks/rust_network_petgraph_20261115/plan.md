# Plan: Rust Network Graph Modernization (`petgraph`)

- [ ] Inventory custom graph storage, adjacency, and traversal routines in `rust/crates/innovate-core/src/network.rs`.
- [ ] Add `petgraph` (matching the existing locked `crates.io` version) directly to `rust/crates/innovate-core/Cargo.toml`.
- [ ] Refactor `network.rs` graph structures and algorithms onto `petgraph` while preserving deterministic arrival/tie-breaking behavior.
- [ ] Verify full Rust (`cargo test --locked`), Python (`pytest tests/native`), and `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc` before review and archive.
