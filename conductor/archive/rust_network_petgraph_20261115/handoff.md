# Track 7 implementation and integration handoff

## Delivered source

Draft [PR #623](https://github.com/edithatogo/innovate/pull/623) implements the
bounded incoming-network graph migration. No merge is authorised in this lane.

- Base: `42189a223fd44c823ca11aec689ea67b11c25790`.
- Durable ownership claim: `226c93856e23df8f7a9d4e7c7da7a42d3ce10610`.
- Tests-first commit: `07d6d897b3f2b57672e6c9a2d4cb54ac1dfcd6e0`.
- Reviewed/tested candidate `network.rs` SHA-256:
  `2e1f90b06d97b813c5038e42ff58c761df6a5ef654d4082e86123bfd49b3b523`.
- Unchanged new integration-test SHA-256:
  `25d86c7e3b89d1c69cdd184879bcbc9b7f513cb96432297551e22854ad611a7c`.
- Unchanged independent fixture SHA-256:
  `57753a47801567e9ae76a71ba7c15926bf57d3a26e5f89694b2722c708ffead1`.

The final published commit and current hosted outcomes are recorded in the PR
handoff. File hashes identify the exact executable source tested here without
requiring a self-referential commit hash inside the commit itself.

`DiGraph<usize, f64>` replaces custom incoming adjacency. Nodes retain original
agent indices; node weights cache incoming degree. Reverse insertion preserves
input-relative floating-point folds. Undirected inputs use paired arcs with
independent per-target normalization. Public JSON, checkpoints and Kairos
scheduling/RNG/resource/queue code are unchanged. No BFS/path algorithm exists
in the original module, so none was invented.

## Executed verification and its boundary

| Evidence | Result | Scope |
| --- | --- | --- |
| Original implementation, existing network target | 37 passed | Actual repository manifest and lock |
| Original implementation, characterization | 8 unit + 5 integration passed | Tests written before migration; ten hand-calculated graph cases |
| Candidate network unit selection | 8 passed | Actual core source with pending direct dependency in external test manifest |
| Candidate integration targets | 37 network + 5 new graph + 5 network diffusion + 17 Kairos passed | Same conditional dependency declaration; 72 selected tests including units |
| Baseline/candidate full-response parity | 160 exact matches | Ten fixed graph cases x sixteen fixed seeds, including 0 and maximum u64 |
| Clippy with warnings denied | Passed | Complete core library and four selected test targets under external manifest |
| Rust formatting and diff whitespace | Passed | Changed Rust files and replay probe |
| Equivalent isolated dependency closure | 101 packages before/after; no added package or feature | Does not assert full workspace feature unification |
| Repository policy, Conductor links, development migration policy | Passed | Python 3.15.2; not installed-consumer/scientific acceptance |
| Actual repository dependency check | Blocked | Shared manifest has no direct `petgraph` declaration |
| Release migration gate | Failed as required | 38 mandatory scientific/acceptance obligations remain open |

Commands, exit codes, source/toolchain/lock identity and actual logs are retained
under `evidence/`. `prepare_test_overlay.py` reproduces an external test package;
it does not edit any shared manifest. `replay_probe.rs` compares separately
compiled baseline/candidate core libraries. This is synthetic migration parity,
separate from the independent graph tables and from empirical validity.

To reproduce while the direct shared dependency is still absent, create a fresh
external package and reconcile **only its copied lock** before using `--locked`.
The initial metadata command intentionally omits `--locked`; it updates no
repository file. Use Rust 1.99.1 and an adequately provisioned private Cargo cache
and target directory (the offline commands require dependencies already cached).

```sh
NETWORK_OVERLAY=/tmp/innovate-network-verification-new
python3 conductor/tracks/rust_network_petgraph_20261115/evidence/prepare_test_overlay.py "$NETWORK_OVERLAY"
cargo metadata --offline --manifest-path "$NETWORK_OVERLAY/Cargo.toml" --format-version 1
cargo test --locked --offline --manifest-path "$NETWORK_OVERLAY/Cargo.toml" --lib network -j 1
cargo test --locked --offline --manifest-path "$NETWORK_OVERLAY/Cargo.toml" --test network --test network_petgraph --test network_diffusion --test kairos -j 1
```

Storage was checked before builds. The nearly full workspace disk was avoided;
this lane used its own temporary checkout, Cargo cache and target directory.
Other workers' data and environments were preserved.

## Exact next actions for the sole integrator

Owner: `6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53`. No acceptance of this request has
been assumed.

1. Add to `rust/crates/innovate-core/Cargo.toml`:

   ```toml
   petgraph = { version = "=0.8.3", default-features = false }
   ```

2. Regenerate the shared `Cargo.lock` so `innovate-core` gains its direct
   `petgraph` dependency. Preserve the already locked 0.8.3 package/checksum:
   `8701b58ea97060d5e5b155d383a69952a60943f0e6dfe30b04c287beb0b27455`.
   No second petgraph version or new feature is requested. This supersedes the
   preliminary unnecessary `std` feature request.
3. Reconcile the shared registry entry to in progress / source implemented,
   integration pending. Keep the track active and do not mark complete or archive.
4. On the resulting exact integration head, run the ordinary locked network
   unit/integration targets, inspect `cargo tree -e features -i petgraph`, then
   the required workspace, installed-consumer, documentation, Miri, security,
   coverage and native mutation gates. Compile-time/runtime overhead remains
   unmeasured; the plan's no-overhead objective is not established.
5. Retain separate release/scientific acceptance and all downstream planning-only
   restrictions. Check current PR checks and reviews before any promotion.

Linalg PR #621 and statistics PR #622 remain their existing owners' work. This
lane changes no shared Cargo file, registry, CI workflow, other production
module, migration obligation, tolerance, permission, deployment or publication
configuration. No release or deletion was performed.
