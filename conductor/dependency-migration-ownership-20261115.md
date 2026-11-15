# Dependency migration ownership and acceptance — 15 November 2026

## Current authority and receipt

Dylan retains sole integration coordinator
`6aeeeda6-d4ec-8320-a3b5-c3e2362e5b53`. The authoritative
[accepted archival handoff](https://github.com/edithatogo/innovate/issues/624#issuecomment-6892039719)
receives the three named worker packets and transfers their residual obligations.
**Acceptance for chat archival is not scientific, release, dependency,
performance, or whole-track qualification.** This coordinator remains active.

This reconciliation replaces stale operational claims in the
[original immutable ownership record](https://github.com/edithatogo/innovate/blob/c620f80e7fc305a155f31092f47f35fb85d9a1ae/conductor/dependency-migration-ownership-20261115.md).
That revision preserves the initial observations, review findings and evidence;
its then-open/preparation/retained-ownership labels are not current assignments.

Initial main inspected in this pass was
`e538af2c9aeb86a439fbeb955e752d1e2bba5b6a`. It advanced through externally
merged #628 to `03fa096d633aa935f7e34575ad2f53f14b44ce6a`. Exact comparison
showed only #628's nine supply-chain/workflow/documentation paths, not this
file or the reviewed network/Bayesian source. This pass did not merge #628
or qualify its security, publication or deployment behavior.

## Reconciled immutable anchors

| Packet | Final PR head | Receiving commit / disposition |
| --- | --- | --- |
| HTTP #618 | `8f30c5a03cc48960b464e8c18a0a750e73b6e2ae` | `349fb0d88b765026385f5361d096070aa2c2795f`; merged, do not redo |
| Tabular #619 | `f2742d3a99952914f1e7193f0d57f67eccbd02ee` | `1cd9af803a21cdb08705cf77a45b27346afa4b69`; merged, do not redo |
| Linalg #621 | `4b2f152d868d901c9f7aa222af3c7d46b3e223c1` | `1e190cfb29be95e0622e79c06bdcb4e272bea1d6`; merged, acceptance hold retained |
| Network #623 | `ead7d5e5a61161dc4f73f5cb508678d3b49952c2` | `5bc63fa0874b402dfa8ee8c69c4a7ef5766735ea`; merged implementation, residual qualification open |
| Statistics #625 | `f89d148e701f844bd6f4c7b13a491ded376bfd82` | `e538af2c9aeb86a439fbeb955e752d1e2bba5b6a`; merged implementation, residual qualification open |
| Coordination #626 | `43201a040888a339e579d34cb72c2b93a6b6ced7` | `c620f80e7fc305a155f31092f47f35fb85d9a1ae`; merged historical record, superseded operational labels below |
| Preserved statistics #622 | `6a85b4c91d5dfc427e17f5fba885681e04ae31ac` | Closed, unmerged; evidence accepted and retained, not a second implementation |
| Original CI #615 | `3d2a6ec638d6d46b2341d9f77e035d2fc44cd72b` | Closed, unmerged; superseded by #616, no residual worker ownership |
| Replacement CI #616 | `7edb822277c8d871396c87b558f8d3bb7a36142f` | `1f9e5436f65ba0c7c3d56036d18ba73aea2ebffd`; existing owner preserved |
| Supply chain #628 | `879ed340bab3759407fd0ae71f4062f7127f1e08` | `03fa096d633aa935f7e34575ad2f53f14b44ce6a`; externally merged, existing owner preserved |

For closed-unmerged PRs, a synthetic `merge_commit_sha` is not a receiving
commit. Merge status and source-tree identity do not establish qualification.
Re-read live main, PRs and #624 before subsequent integration or path claims.

## Worker chat archival decisions

| Worker chat | Accepted durable handoff | Disposition |
| --- | --- | --- |
| Network `6af950b7-c300-83ec-9bb0-316f9a9718a3` | [#624 comment 6890788042](https://github.com/edithatogo/innovate/issues/624#issuecomment-6890788042), source/evidence `ff7e077a672721bc14fe045454164766f3fa0e45` | May archive; implementation packet received, residual qualification transferred to NET-QUAL below |
| Preserved statistics `6af95062-c3bc-83ec-a013-841a9bc3ab02` | [#626 comment 6890860269](https://github.com/edithatogo/innovate/pull/626#issuecomment-6890860269), packet `6a85b4c91d5dfc427e17f5fba885681e04ae31ac` | May archive; superseded/preserved packet received, findings transferred to STAT-NUM below; separate #625 source owner not released |
| Original CI `6af8f5be-1c38-83ec-90ba-18b2e1687946` | [#615 comment 6891713298](https://github.com/edithatogo/innovate/pull/615#issuecomment-6891713298) and [#624 comment 6891719758](https://github.com/edithatogo/innovate/issues/624#issuecomment-6891719758) | May archive; explicit release accepted, no residual implementation/workflow/registry/branch-maintenance/cleanup/monitoring claim |

These three chats retain no further implementation or monitoring duty. Their
source, fixtures, patches, logs and immutable history must be preserved; chat
archival does not authorise branch pruning or evidence deletion. The receipt
does not itself change ChatGPT archive state. No additional chat is cleared.

## Residual obligations and their durable owners

### NET-QUAL — Track 7 / #623 / #624

The coordinator receives remaining integration and qualification from the
network worker. Its immutable
[handoff](https://github.com/edithatogo/innovate/blob/ff7e077a672721bc14fe045454164766f3fa0e45/conductor/tracks/rust_network_petgraph_20261115/handoff.md)
and [review record](https://github.com/edithatogo/innovate/blob/ff7e077a672721bc14fe045454164766f3fa0e45/conductor/tracks/rust_network_petgraph_20261115/evidence/review.md)
were inspected. `network.rs` has identical Git blob
`364068cb7171bf7cd65d5aa6189371c1d64f66a2` at that worker head and inspected
main `e538af2c9aeb86a439fbeb955e752d1e2bba5b6a`.

The recorded 72 selected tests and 160 full-response comparisons remain
worker-executed original-base evidence using an external dependency manifest,
not new integrated-head runs. Independent graph fixtures, deterministic
ordering, directed/undirected normalization, isolates and Kairos boundaries
remain part of the accepted packet, not empirical validity claims.

The current core manifest supplies `petgraph = "=0.8.3"`; the missing direct
declaration is no longer the source-state blocker. It does not use the requested
`default-features = false`. Exact feature/lock-closure reconciliation remains
with the coordinator, along with applicable integrated workspace, installed
Python/Julia/R/platform, documentation, Miri, coverage, native mutation,
security and measured runtime/memory/compile-overhead acceptance. No zero-overhead
or full-workspace-closure claim is accepted.

### STAT-NUM — Track 5 / #625 / immutable #622 packet / #624

The coordinator owns reconciliation with the existing #625 source owner;
the documentation owner retains substantive source/claim review. The preserved
[packet README](https://github.com/edithatogo/innovate/blob/6a85b4c91d5dfc427e17f5fba885681e04ae31ac/conductor/tracks/rust_stats_distributions_20261115/evidence/ownership-handoff-20261115/README.md)
and [qualification receipt](https://github.com/edithatogo/innovate/blob/6a85b4c91d5dfc427e17f5fba885681e04ae31ac/conductor/tracks/rust_stats_distributions_20261115/evidence/ownership-handoff-20261115/qualification.json)
were read. Exact comparison from baseline
`b344f379926e934be30edc0842779dc5e3455f86` to evidence head
`6a85b4c91d5dfc427e17f5fba885681e04ae31ac` confirms one commit adding exactly
20 handoff files and no other change. Its completed migration remains a deferred
patch, not a second applied production implementation.

Preserve each finding and its original execution scope:

- The independent 83-row oracle and reconstructed candidate result:
  **393 passed, one failed, one inherited live-OECD ignore** across 28 targets.
  **14 of 36 CDF/survival comparisons fail unchanged relative tolerance
  `2e-13`**. The 22 inverse-CDF and 18 normalized-log-PDF cases pass in that
  candidate receipt; none is a current-main execution.
- The negative control for normalized-log-density subtraction loses the
  `-5e-21` near-centre Gaussian score. Reconcile relative/unnormalised score
  semantics against the merged source; do not dismiss it on supersession.
- The demonstrated 80-draw constant-chain ESS regression, exact-split constancy
  guards, other constant-chain cases and preserved R-hat overflow errors.
  Empirical type-7 bootstrap intervals, null/count behavior and RNG remain
  constraints; no analytic helper needed replacement in bootstrap.
- Scientific-review freshness failed for the preserved candidate. Substantive
  review of the accepted Bayesian source remains required; a refreshed shared
  fingerprint does not independently establish scientific qualification.

These results do not test merged #625 or current main. Their exact integrated-head
disposition remains open. Do not weaken tolerances, ignore failing cases, replace
independent oracle values or reopen/reimplement #622. The current manifest uses
`statrs = "=0.19.1"`, not the requested no-default-features/std-only declaration;
feature, lock and advisory qualification remain coordinator-owned. The original
base dependency patch must not overwrite the newer shared lock.

### CI-CURRENT — coordinator's shared-CI intake in #624

The #615 ownership release is accepted, rather than merely awaiting acknowledgement.
Shared follow-ups remain here with existing #616/#628, documentation and security
owners preserved. The #615 worker has no branch-maintenance or monitoring duty.

Retain exact-current-head gate qualification and any controlled performance
measurement. #615's own cancelled CI/coverage are not passes; neither cache
success nor different-revision timings establish attributable hosted speedup.
The #622 Release Drafter missing-configuration/fallback-404 observation remains
an unresolved reconciliation item: run `43688357924`, job `132733491111`, at
its historical evidence SHA. It is not asserted as a current-main failure.
This pass did not rerun workflows or qualify the latest main.

## Shared ownership, dependencies and unchanged holds

The coordinator retains all shared Cargo manifests/locks, common Conductor
registry, shared CI integration, interfaces and serial `causal.rs`. Existing
specialist ownership is preserved, except the explicit #615 release above.
Statistics must not concurrently edit the linalg causal slice.

**HTTP + tabular precede linalg/statistics/network. Accepted linalg precedes ODE
and regime. Accepted statistics precedes NUTS, with a serial `bayesian.rs`
handoff.** No merged implementation or archived plan lifts those holds.

- Linalg remains with its existing owner and
  [review 5909490616](https://github.com/edithatogo/innovate/pull/621#pullrequestreview-5909490616).
  The finite `1e75` trial penalty/final-bound concern and changed or detached
  termination semantics are unresolved source-level findings, not locally
  reproduced failures in this coordinator chat. ODE/regime acceptance stays held.
- Documentation remains with the existing #613 owner, including
  [browser handoff 6890734373](https://github.com/edithatogo/innovate/pull/613#issuecomment-6890734373)
  and substantive Bayesian source review. Historical browser run `43685509160`
  executed 141 passes and one artifact-identity failure out of 142; no disposition
  is inferred from later unrelated merges.
- #559/Voiage ownership remains. Offline-security #590 and its existing owner
  retain dedicated runner, local assets and actual scanner/agent qualification;
  a queued job or merged caller is not acceptance.
- Track 3 Kairos publication remains an external, separately authorised
  prerequisite. Upstream is read-only here; retain the existing pin. #628 does
  not authorise account choices, publication, runner substitution or blanket
  registry-only supply-chain acceptance.
- All **38 scientific obligations**, four human scenario checks, fail-closed
  release/security gates and planning-only consumers, including Tobac, remain.
  No new worker, publication, permissions change, upstream write, paid compute
  expansion, blanket merge/pruning or disposal of unverified work is authorised.

Each later integration handoff must supply immutable head/base SHAs, exact paths,
shared changes, independent tests, real commands/exits/toolchain, hosted checkout
identity, skips, review findings and limitations. Reconcile changed heads/bases;
merge only qualified packets sequentially and verify receiving SHA/tree and
applicable receiving-main checks. Generic review or CI is not scientific sign-off.

## Registry interpretation and verification limits

`conductor/tracks.md` is not edited by this receipt pass. In particular, historical
HTTP/tabular planned labels do not call for reimplementation, linalg completion
labels do not lift its acceptance hold, and merged statistics/network source does
not complete scientific qualification. Their current operational ownership and
residual obligations are given above and in the accepted #624 receipt.

Fresh checks in this pass were GitHub PR/ref reads, immutable evidence reads,
GitHub comparisons and source blob-identity inspection. The sandbox capacity
probe failed before command execution. No local build, test, disk-capacity result,
checksum recomputation, numerical reproduction, new independent review or fresh
integrated-head qualification is claimed. This one-file documentation correction
requires its own review/checks before merge; the published #624 acknowledgement
already establishes accepted handoff and archival eligibility.
