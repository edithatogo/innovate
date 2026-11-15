# Rust Network Graph Modernization (`petgraph`)

Status: in progress; graph characterization and bounded migration in draft PR #623.
Shared dependency acceptance and complete integration gates remain pending.

- [Specification](spec.md)
- [Implementation plan](plan.md)
- [Ownership and exact dependency handoff](ownership.md)

Priority: P2. `petgraph` 0.8.3 is already locked. The integrator must verify the
resolved dependency/feature closure; compile-time overhead remains unmeasured.
