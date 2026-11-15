# Plan: Rust Network Graph Modernization (`petgraph`)

> **Execution Order:** #7 (Self-contained in `network.rs`; parallelizable with #4–#6 after Phase A tracks #1–#2 complete).

## Phase 1: Graph Inventory, Dependency Pin & Characterization Tests
- [x] **1.1 Inventory `network.rs` graph structures:** Reconciled live main `42189a223fd44c823ca11aec689ea67b11c25790`; inventory and independent read-only review recorded in [ownership](ownership.md) and [specification](spec.md). No BFS/DFS/path/centrality routine exists.
- [!] **1.2 Pin `petgraph` directly:** Sole integrator owns manifests/lock; exact `=0.8.3`, `default-features = false`, `features = ["std"]` request is in [ownership](ownership.md) and draft PR #623. Dependency acceptance and locked feature-tree evidence remain pending.
- [x] **1.3 Add deterministic characterization tests:** Five private tests, ten independent graph tables and five public tests added before migration. Existing integration target: 37 passed. Network unit selection: 8 passed (three existing); new public target: 5 passed. See [baseline log](evidence/baseline-network.log), [characterization log](evidence/baseline-characterization.log) and paired JSON source-hash receipts. Expected values are hand-calculated, not production snapshots.
- [x] **1.4 Conductor Review (Phase 1):** Independent read-only review found no blocking test/fixture defect; remaining plan wording and timestamp corrected. [Review record](evidence/review.md). Shared dependency 1.2 remains blocked; this is scoped characterization acceptance, not full Phase 1 integration acceptance.

## Phase 2: Refactor `network.rs` onto `petgraph`
- [~] **2.1 Replace custom graph storage:** Use `petgraph::graph::DiGraph` with explicit population nodes and paired arcs for undirected input; preserve destination-specific normalized weights and original incoming order. Public JSON and checkpoints stay unchanged.
- [ ] **2.2 Replace incoming-incidence/degree access:** Use petgraph incoming edges and `visit::EdgeRef` while preserving exact weighted/unweighted arithmetic and cached degree lookup. BFS/DFS/components/paths/centrality are inapplicable because the module has none. Run focused network contracts without changing any scheduler or RNG semantics.
- [ ] **2.3 Conductor Review (Phase 2):** Run `conductor-review` on Phase 2 changes, apply any fixes, re-run `cargo test -p innovate-core --locked network`, and commit checkpoint.

## Phase 3: Full Workspace, Polyglot & Gate Verification
- [ ] **3.1 Run full Rust & Python test suites:** Execute `cargo test --workspace --locked` and `pytest tests/native`.
- [ ] **3.2 Run docs, `Miri` & security gates:** Execute `RUSTDOCFLAGS="--cfg docsrs -D warnings" cargo doc --no-deps --workspace`, scoped `cargo +nightly miri test -p innovate-core --lib network`, and `cargo audit`.
- [ ] **3.3 Conductor Review (Phase 3 & Track Sign-off):** Run `conductor-review`, apply any fixes, re-run verification gates, and mark track complete in `conductor/tracks.md`.
