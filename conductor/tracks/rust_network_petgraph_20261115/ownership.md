# Track 7 network migration ownership and integration request

## Bounded assignment

Dylan authorised this lane on 15 November 2026: `network.rs`, new dedicated
deterministic graph tests/fixtures, and this track's evidence only. The checked
base is `42189a223fd44c823ca11aec689ea67b11c25790`.

Live reconciliation found HTTP PR #618 and tabular PR #619 merged. Linalg PR
#621 remains open at `4b2f152d868d901c9f7aa222af3c7d46b3e223c1`; its changed
paths do not include this lane. The open-PR search returned #621 only and the
network branch search returned no branch. This records the observed state, not
authority over any other writer. The statistics owner
`6af95062-c3bc-83ec-a013-841a9bc3ab02` retains Bayesian/bootstrap work.

Claimed paths:

- `rust/crates/innovate-core/src/network.rs`
- `rust/crates/innovate-core/tests/network_petgraph.rs` (new)
- `rust/crates/innovate-core/tests/fixtures/network_petgraph_20261115.json` (new)
- `conductor/tracks/rust_network_petgraph_20261115/**`

The sole integrator `6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53` retains all shared
Cargo manifests/locks, registry, CI and integration. No merge, release,
publication, permission change, downstream implementation or deletion is part
of this assignment. Scientific qualification and planning-only restrictions
remain in force.

## Exact shared dependency request

Please add this direct dependency to `[dependencies]` in
`rust/crates/innovate-core/Cargo.toml`:

```toml
petgraph = { version = "=0.8.3", default-features = false, features = ["std"] }
```

The checked `Cargo.lock` already contains crates.io `petgraph` 0.8.3, checksum
`8701b58ea97060d5e5b155d383a69952a60943f0e6dfe30b04c287beb0b27455`.
Regenerate the shared lock with Cargo so `innovate-core` records the direct
dependency; retain the existing package/version/checksum and do not add a second
petgraph version. `graph::DiGraph`, incoming-edge iterators and `visit::EdgeRef`
need no optional graph algorithms, serialization, parallelism or parser features.
Verify the resolved features and package closure with
`cargo tree --locked -p innovate-core -i petgraph` and
`cargo tree --locked -p innovate-core -e features -i petgraph`.

Shared registry status and full workspace/installed-consumer/docs/Miri/security
gates also remain with the integrator. This lane will supply exact source-bound
receipts and report an unaccepted dependency request as a blocker, not a pass.

## Inventory correction before implementation

At the base revision `network.rs` has one custom incoming adjacency list,
per-destination normalized weights/totals, and neighbor folds for exposure.
It has no BFS, DFS, connected-components, shortest-path or centrality routine.
Service, scheduling, clock, queue and RNG remain owned by Kairos. Track tasks
referring to nonexistent algorithms do not authorise inventing them.

The arithmetic contract preserves **input edge order per destination**. A
directed petgraph graph traverses the most recently inserted incident edge first;
reverse insertion is therefore needed to preserve the old summation order.
Undirected input requires two directed arcs because its two destination-specific
normalizations can differ. Every population index, including isolates, must
receive a graph node. Do not sort input edges or serialize petgraph internals
into checkpoints.

Primary API evidence: [petgraph 0.8.3 Graph documentation](https://docs.rs/petgraph/0.8.3/petgraph/graph/struct.Graph.html#method.edges_directed).

## Execution environment

The initial storage check reported approximately 6 MB free on `/workspace` and
4 GB on `/tmp`. A dedicated checkout and bounded build output use `/tmp`.
Other workers' files, environments and build caches are preserved. Verification
must record any temporary dependency overlay separately from a repository
`--locked` build; it cannot qualify shared integration or release acceptance.
