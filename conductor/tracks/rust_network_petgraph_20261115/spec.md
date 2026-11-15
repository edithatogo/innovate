# Specification: Rust Network Graph Modernization (`petgraph`)

## Objective
Promote `petgraph` (already present transitively in `Cargo.lock` via `kairo-ecs-*`) to a direct dependency in `rust/crates/innovate-core/Cargo.toml` and replace custom adjacency/traversal/topology code in `rust/crates/innovate-core/src/network.rs` with `petgraph::graph` and `petgraph::algo`.

## Scope
- Replace custom graph adjacency lists, BFS/DFS traversals, connected components, and shortest-path calculations in `network.rs` with `petgraph`.
- Preserve deterministic node/edge ordering, deadline/waiter tie-breaking, and all `network::tests` and `tests/native` contracts.
- Add zero new transitive dependencies or compile-time overhead.
