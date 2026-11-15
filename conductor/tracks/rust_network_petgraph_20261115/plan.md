# Plan: Rust Network Graph Modernization (`petgraph`)

> **Execution Order:** #7 (Self-contained in `network.rs`; parallelizable with #4–#6 after Phase A tracks #1–#2 complete).

## Phase 1: Graph Inventory, Dependency Pin & Characterization Tests
- [ ] **1.1 Inventory `network.rs` graph structures:** Read `rust/crates/innovate-core/src/network.rs` and list all custom adjacency lists, BFS/DFS traversals, shortest-path/centrality helpers, and deterministic node-ordering assumptions in `spec.md`.
- [ ] **1.2 Pin `petgraph` directly in `rust/crates/innovate-core/Cargo.toml`:** Add `petgraph` matching the already-locked `crates.io` version (`0.8.3` or current `Cargo.lock` version) so no duplicate `petgraph` versions are introduced. Verify with `cargo tree -p innovate-core -i petgraph`.
- [ ] **1.3 Add deterministic tie-breaking characterization tests:** Ensure `rust/crates/innovate-core/src/network.rs` unit tests lock exact node arrival order, tie-breaking on equal edge weights/distances, and disconnected component handling. Run `cargo test -p innovate-core --locked network`.
- [ ] **1.4 Conductor Review (Phase 1):** Run `conductor-review` on Phase 1 changes, apply any fixes, re-run `cargo test -p innovate-core --locked network`, and commit checkpoint.

## Phase 2: Refactor `network.rs` onto `petgraph`
- [ ] **2.1 Replace custom graph storage with `petgraph::graph::DiGraph` / `UnGraph`:** Migrate internal node/edge representation in `rust/crates/innovate-core/src/network.rs` while keeping public JSON/struct contracts unchanged.
- [ ] **2.2 Replace custom traversal/path routines:** Migrate BFS/shortest-path/degree calculations to `petgraph::visit` and `petgraph::algo`, preserving deterministic tie-breaking by node index/ID. Run `cargo test -p innovate-core --locked network`.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked network`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native`.
- [ ] **3.2 Run docs, `Miri` & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, scoped `cargo +nightly miri test -p innovate-core --lib network`, and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
