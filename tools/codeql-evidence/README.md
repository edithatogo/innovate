# Historical Rust analysis contexts

The two standalone Cargo manifests beside the archived replay and scenario
probes give CodeQL's Rust analyzer real semantic contexts for those existing
files. They are separate workspaces: they do not change the production
workspace, its lockfile, scientific acceptance or release status. The probes
and their recorded receipts are unchanged.

The scenario context pins historical Innovate commit
`3681c8a820e7f3f852ff7429687ba3997cbcae3f` and retains the original
hashbrown patch. Its lock was derived from the historical lock; the extra
entries identify the analysis package and the same core source as a Git
package.

The replay context uses baseline commit
`07d6d897b3f2b57672e6c9a2d4cb54ac1dfcd6e0` and the exact candidate source
from `ff7e077a672721bc14fe045454164766f3fa0e45`. The candidate is exposed as
an analysis-only Cargo package by support commit
`b0a60804f6bcd407e2548197301a3976011b08e1`, whose sole change is a standalone
manifest matching the recorded `petgraph` overlay. Its `network.rs` SHA-256 is
`2e1f90b06d97b813c5038e42ff58c761df6a5ef654d4082e86123bfd49b3b523`.
The retained branch `codex/innovate-codeql-historical-candidate` keeps that
commit reachable; do not delete it while the replay lock depends on it.

To check the analysis contexts without executing either historical probe:

```sh
cargo check --locked --manifest-path conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.toml --bin replay-probe
cargo check --locked --manifest-path conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.toml --tests
cargo audit --file conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.lock
cargo audit --file conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.lock
```

The original evidence explains how the historical runs were produced. These
analysis contexts do not certify replay equivalence, independent scientific
validation, publication or the separate offline-security lane.
