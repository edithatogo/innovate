# Dependency migration ownership and acceptance — 15 November 2026

## Authority and interpretation

Dylan retains the continuing ordinary Pro ChatGPT conversation as the sole
Innovate integration coordinator. The live coordination thread is
[issue #624](https://github.com/edithatogo/innovate/issues/624).
This record is a timestamped reconciliation, not scientific certification,
permission to publish, or permission to replace an existing worker.

The coordinator owns shared Cargo manifests and locks, common Conductor registry,
shared CI integration, cross-lane interfaces and `causal.rs`. Existing specialist
documentation, issue #559/Voiage, CI-optimisation and offline-security owners
remain in place. Their existing proposals must be reviewed and integrated once,
not overwritten by a parallel implementation.

Re-read live main, PRs, this record's updates in #624 and the owning track before
claiming a path. An unchecked box, archived plan, empty PR search, passing status
or merged PR does not independently establish ownership or acceptance.

## Immutable reconciliation anchors

| Packet | Final PR head | Receiving commit | Interpretation |
| --- | --- | --- | --- |
| HTTP Track 1, #618 | `8f30c5a03cc48960b464e8c18a0a750e73b6e2ae` | `349fb0d88b765026385f5361d096070aa2c2795f` | Merged implementation; reuse, do not redo. |
| Tabular Track 2, #619 | `f2742d3a99952914f1e7193f0d57f67eccbd02ee` | `1cd9af803a21cdb08705cf77a45b27346afa4b69` | Merged implementation; reuse published-Polars work and preserve Arrow contracts. |
| Linalg Track 4, #621 | `4b2f152d868d901c9f7aa222af3c7d46b3e223c1` | `1e190cfb29be95e0622e79c06bdcb4e272bea1d6` | Merged during coordinator review; downstream acceptance held. |
| Statistics Track 5, #625 | `a39f81ca318c6ec98ac6ea3a62f9c99448dd0a21` | None at inspection | Open proposal on base `1e190cfb29be95e0622e79c06bdcb4e272bea1d6`; not accepted. |

Initial main was `42189a223fd44c823ca11aec689ea67b11c25790`. While the
coordinator reviewed #621, another actor merged it at 06:38:32 UTC on 15 November.
This coordinator did not invoke that merge. Main was subsequently read back as
`1e190cfb29be95e0622e79c06bdcb4e272bea1d6`.

The #621 PR test merge `711ec65faed873ad661a3853e36fbd92c1ebff0c` and its
receiving commit have the identical Git tree
`e965faac4dce04bf85bd164abb59860c16201559`. This establishes tree identity,
not complete qualification or correctness.

Existing owner anchors, without expanding their acceptance claims:

- Documentation #613 merged as `42189a223fd44c823ca11aec689ea67b11c25790`.
- Issue #559/Voiage #614 merged as `785c0451ff05f978675bb93150dddbda569e521c`.
- CI optimisation #616 merged as `1f9e5436f65ba0c7c3d56036d18ba73aea2ebffd`;
  preserve related #615 ownership as well.
- Offline security #590 merged as `1e06bcc33f47b74600e842bd7b3fe13fcd2dbd84`.
  Its stale draft warning is not proof of present PR state; its merge is also
  not proof that a dedicated runner, offline assets or real scanner/agent
  integration have been qualified. Preserve the existing security owner/gates.

## Lane map and dependency order

**HTTP + tabular precede linalg, statistics and network. Accepted linalg precedes
ODE and regime. Accepted statistics precedes NUTS.** Independent cleared tests
and adapter slices can proceed without claiming dependent release qualification.

| Track | Existing or proposed lane | Write boundary and prerequisite |
| --- | --- | --- |
| 1 HTTP: `rust_http_ureq_20261115` | Existing merged owner, #618 | No duplicate HTTP migration. Hand shared follow-ups to the coordinator. |
| 2 tabular: `rust_tabular_crates_io_20261115` | Existing merged owner, #619 | No duplicate tabular migration. Preserve portable table/Arrow contracts. |
| 4 linalg: `rust_linalg_optimization_20261115` | Existing #621 owner, including bounded review follow-up | `numerics.rs` and cleared dedicated tests/evidence. Existing shared hunks are retained as proposals for serialized integration. ODE/regime acceptance remains held. |
| 5 statistics: `rust_stats_distributions_20261115` | Existing proposal #625; do not start a second statistics writer | `bayesian.rs`, `bootstrap.rs`, cleared dedicated tests/evidence. No concurrent `causal.rs`, manifests, locks, registry, CI or common-doc edits. HTTP/tabular first. |
| 7 network: `rust_network_petgraph_20261115` | Dylan is preparing a worker, subject to live ownership clearance | `network.rs`, cleared dedicated tests/evidence. Preserve scheduling, deterministic order, replay and Kairos boundaries. HTTP/tabular first; no upstream writes. |
| 6 ODE: `rust_ode_integration_20261115` | No new writer assigned by this coordinator | Wait for accepted Track 4 interfaces and exact receiving revision. |
| 9 regime: `rust_regime_changepoint_20261115` | No new writer assigned by this coordinator | Wait for accepted Track 4 interfaces and exact receiving revision. |
| 8 NUTS: `rust_nuts_hmc_20261115` | No new writer assigned by this coordinator | Wait for accepted Track 5 and an explicit serial `bayesian.rs` handoff. |
| 3 Kairos crates.io: `rust_kairos_crates_io_20261115` | External publication prerequisite, not authorised here | Upstream is read-only. Do not publish or change the existing pin without separately qualified integration. |
| 10 supply chain: `ci_supply_chain_hardening_20261115` | Existing CI/security owners; shared changes received here | No new account, permission, publication or runner choices. Strict registry-only closure cannot be asserted while Track 3 is unresolved. |

### Serial `causal.rs` reservation

`rust/crates/innovate-core/src/causal.rs` belongs to the coordinator's serial
integration boundary. Retain the already-proposed linalg changes from #621;
qualify that slice first. Statistics must submit a separate proposed causal
patch, not edit the file in parallel. Only after the linalg acceptance revision
is pinned may a separately reviewed statistics causal slice be applied.

### Shared-file overlap discovered in #625

At head `a39f81ca318c6ec98ac6ea3a62f9c99448dd0a21`, #625 proposes changes to
`Cargo.lock`, the core manifest, `conductor/tracks.md` and the documentation
review fingerprint. Those proposals are retained, not independently rewritten.
Its changed-file list does not include `causal.rs` or `bootstrap.rs`.
Absence of a bootstrap change is not automatically a defect: the owner must
state whether its requirements are retained, completed or still open.

The coordinator therefore narrowed this repository packet to this new file
only, avoiding a competing registry edit. #624 and the comments on #621/#625
record the operational reservations. Shared registry corrections must be
applied once during integration, after reconciling the active proposal.

## Registry reconciliation, not retrospective erasure

At receiving commit `1e190cfb29be95e0622e79c06bdcb4e272bea1d6`,
`conductor/tracks.md` still labels Tracks 1 and 2 planned, while Track 4 is marked
completed and linked to its archive. Reconcile those entries as follows:

| Existing label | Reconciled meaning |
| --- | --- |
| HTTP planned | Implementation merged in #618; no redo. Preserve specific qualification limits. |
| Tabular planned | Implementation merged in #619; no redo. Preserve specific qualification limits. |
| Linalg completed/archive | Implementation merged in #621; coordinator acceptance and dependent ODE/regime integration held on the findings below. Keep the historical archive as evidence. |
| Statistics planned, proposed archival in #625 | Existing owner/PR received, not accepted. Resolve actual scope and exact-head evidence before completing the track. |
| Network planned | Worker preparation is authorised but exact paths must first be checked against live ownership. |

This record does not silently change historical metadata, approve new consumer
implementation, or declare any scientific obligation complete.

## #621 qualification findings and evidence

Exact-head coordinator review:
[review 5909490616](https://github.com/edithatogo/innovate/pull/621#pullrequestreview-5909490616).
This is a coordinator review, not an independent reviewer or an executed local
numerical validation.

Two important source-level findings require a bounded owner follow-up:

1. Invalid/out-of-bound trial residuals are replaced by finite `1e75` values.
   Legitimate residuals can exceed that penalty, so it is not a guaranteed
   rejection. Final candidate re-evaluation also lacks the declared parameter
   bound check. The review provides a scale-stress regression proposal; it has
   not been executed in this chat.
2. The new library's normalized orthogonality, initial scaled step bound and
   function-evaluation budget differ from the old adapter's criteria. The code
   can also return an initial-point fallback while retaining a candidate's
   termination classification. Qualify the actual semantics and attach
   convergence to the returned valid point; do not infer equivalence from
   matching numeric configuration constants.

Observed hosted evidence at this head:

| Evidence | Observed result / limit |
| --- | --- |
| CI `43685509133` | Workflow reports success; this alone does not dispose of the review findings. |
| Documentation `43685509160`, browser job `132725050345` | 142 cases executed: 141 passed, 1 failed. Mobile dark `/innovate/native-migration/` failed the internal artifact-identity assertion; response for `/innovate/` differed from built `index.html`. Not established as pre-existing or harmless. |
| Browser diagnostics | Artifact `14150317226`; build evidence artifact `14150042950`. Documentation owner retains the repair scope. |
| Offline security `43685509599`, job `132725051187` | Queued with no executed steps when inspected. Not scanner/agent runtime qualification. |
| Release Readiness `43685509288` | Failure observed; no waiver or release claim. |

Receiving-main workflow evidence is a separate acceptance item. A later owner
receipt must identify the actual run, attempt, checkout revision and tree;
reuse of an older branch result is not sufficient for a changed tree.

## Exact-SHA handoff and sequential acceptance

Each owner hands off the PR, full immutable head/base SHAs, exact changed paths,
shared-file proposal, independent expected-result and negative tests, actual
commands/toolchain/exit codes, hosted run/job identities, skips/unexecuted gates,
review findings/fixes and downstream restrictions. A moving branch is not a
final handoff.

The coordinator reviews the full scoped diff and all affected contracts,
checks the tested synthetic merge and base identity, and qualifies applicable
Rust, installed Python, Julia/R/platform, formatting/docs, coverage, policy,
advisory/privacy/security and consumer evidence. New head/base changes require
reconciliation. Merge only one accepted packet at a time; verify the receiving
SHA/tree and receiving-main checks before admitting dependent integration.

All **38 scientific migration/acceptance obligations**, the fail-closed release
gate, and four historical scenario human checks remain. Planning-only consumer
restrictions remain, including Tobac. CI does not constitute empirical,
clinical, publication or human acceptance. Offline-security runtime acceptance
cannot be replaced by a static check or runner substitution.

No release/publication, upstream write, account/permission choice, paid compute
expansion, blanket merge/pruning or disposal/publication of preserved local
stash/ignored material is authorised. This coordinator has not created workers.

## Execution limitation

The ordinary ChatGPT sandbox terminal and Python failed at the infrastructure
layer before command execution. A subsequent terminal/disk-capacity probe also
failed. No local build, test, disk-capacity result or numerical reproduction is
claimed. No local Codex/Work or user's machine was used. Hosted observations
above are separate from proposed regression work and pending qualification.
