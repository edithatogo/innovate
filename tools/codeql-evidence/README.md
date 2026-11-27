# Historical Rust analysis contexts

CodeQL analyzes the network replay and scenario producer in their standalone
Cargo workspaces. The probes and retained fixtures are byte-identical to the
earlier contexts. The dependency backports use the maintained Jiff calendar
and Kairos entity-state adapter already qualified in the current Innovate
core. Each context retains its original Kairos revision and scheduler. The
unused historical scenario hashbrown patch is preserved in the original
manifest. [Chrono advisory](https://rustsec.org/advisories/RUSTSEC-2026-0543.html)

| Context | Original source | Dependency backport |
| --- | --- | --- |
| Network baseline | `07d6d897b3f2b57672e6c9a2d4cb54ac1dfcd6e0` | `0eda3d0670116348fcdc9b3241ea21ad6b66ec3e` |
| Network candidate | `b0a60804f6bcd407e2548197301a3976011b08e1` | `62f2210df27c956795118ab4f417239d0079c225` |
| Historical scenario | `3681c8a820e7f3f852ff7429687ba3997cbcae3f` | `07244129d41605892a69e51f0b52fa1754e2f677` |

Keep the `codex/innovate-secure-network-baseline-20261127`,
`codex/innovate-secure-network-candidate-20261127` and
`codex/innovate-secure-scenario-replay-20261127` branch tips reachable while
the active workspaces use these pins. The original
`codex/innovate-codeql-historical-candidate` branch preserves the earlier
network candidate package and its source from `ff7e077a672721bc14fe045454164766f3fa0e45`.

## Original evidence

Commit `cc2e4da4af9c848043b31f1191c9d96212048276` preserves both original
manifest and lockfile pairs byte-for-byte. The existing replay outputs,
fixture hashes, and scientific receipts retain their original source
and runtime identities.

| Frozen files | Original lock SHA-256 |
| --- | --- |
| [Network manifest](https://github.com/edithatogo/innovate/blob/cc2e4da4af9c848043b31f1191c9d96212048276/conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.toml), [network lock](https://github.com/edithatogo/innovate/blob/cc2e4da4af9c848043b31f1191c9d96212048276/conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.lock) | `60f50310c91b482f2e4284b3ef3038174dbd9afe401f4b399f06008ccc40b3e3` |
| [Scenario manifest](https://github.com/edithatogo/innovate/blob/cc2e4da4af9c848043b31f1191c9d96212048276/conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.toml), [scenario lock](https://github.com/edithatogo/innovate/blob/cc2e4da4af9c848043b31f1191c9d96212048276/conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.lock) | `ab36dbb0404f2262f97e0231b0e08702300d442272e09d75d1866186e7e127bb` |

## Validate the active analysis contexts

```sh
cargo check --locked --manifest-path conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.toml --bin replay-probe
cargo check --locked --manifest-path conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.toml --tests
cargo audit --file conductor/archive/rust_network_petgraph_20261115/evidence/Cargo.lock
cargo audit --file conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/Cargo.lock
```
