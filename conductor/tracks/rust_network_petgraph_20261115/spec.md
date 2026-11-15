# Specification: Rust Network Graph Modernization (`petgraph`)

## Objective
Promote `petgraph` (already present transitively in `Cargo.lock` via `kairo-ecs-*`) to a direct dependency in `rust/crates/innovate-core/Cargo.toml` and replace custom adjacency/traversal/topology code in `rust/crates/innovate-core/src/network.rs` with `petgraph::graph` and `petgraph::algo`.

## Scope
- Replace custom graph adjacency lists, BFS/DFS traversals, connected components, and shortest-path calculations in `network.rs` with `petgraph`.
- Preserve deterministic node/edge ordering, deadline/waiter tie-breaking, and all `network::tests` and `tests/native` contracts.
- Add zero new transitive dependencies or compile-time overhead.

## Source-characterized scope — 15 November 2026

The checked baseline is `42189a223fd44c823ca11aec689ea67b11c25790`.
`Graph::new` owns a `Vec<Vec<(usize, f64)>>` of incoming incidences and cached
per-node totals. The adoption hazard folds those incidences. No BFS, DFS,
connected-components, shortest-path or centrality implementation exists in this
module; the corresponding original plan wording is inapplicable here.

Contract requirements:

- Create every population node in index order, including isolates.
- `[source, target]` exposes the target in directed mode. Opposite directed arcs
  are valid; undirected reverse pairs are duplicates. Reject self-loops,
  repeated edges and out-of-range indices with the existing message.
- Validate weight length and finite strictly positive domain before edge
  validation. Explicit empty weights retain weighted metadata on an empty graph.
- Preserve input-relative incoming-edge order, including floating-point fold
  order. This is not a node-ID sorting contract.
- Normalize by each destination's maximum incoming weight, then accumulate in
  that order. Undirected incidences normalize independently. Preserve positive
  weight underflow to zero and zero imitation for isolates.
- Preserve weighted/unweighted hazard arithmetic, susceptible-agent iteration,
  and the rule that pending agents do not transmit.
- Scheduled graph replacement and checkpoint reconstruction use the same
  constructor. A replacement without weights is unweighted. No serialized
  graph internals or checkpoint/JSON schema change is required.
- Keep all Kairos event, deadline, priority, FIFO, retry, resource and RNG
  control flow unchanged.

The bounded representation is a private directed petgraph graph, using paired
arcs for undirected input and reverse insertion to recover original incoming
iteration order. Cached degree/totals may remain to avoid extra hazard scans.
No new graph algorithm or simulation engine is part of this change.

The integrator owns dependency promotion and feature/lock reconciliation; see
[the exact shared request](ownership.md). Existing package presence is not
measured compile-time evidence. Zero additional package versions and unchanged
feature closure require actual dependency resolution; compile-time overhead
and complete scientific/release gates remain unqualified until measured.
