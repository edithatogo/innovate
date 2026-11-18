# Manuscript source reassessment after #656

Date: 16 November 2026. Decision: **retain an unapproved preparation draft at the
newly assessed source; do not promote scientific, release or submission status**.

The prior edition described `69b674814513e09a683279c679b1584af8bb4214`. Main advanced
to [`57d2916586d107c8affe90355e453abe27f01c26`](https://github.com/edithatogo/innovate/commit/57d2916586d107c8affe90355e453abe27f01c26)
when #656 merged at 11:10:35 UTC on 16 November 2026, according to GitHub's PR
`merged_at` field. The commit's committer timestamp is 11:10:34 UTC; it identifies
a separate event from the PR merge timestamp. The integration worktree was
observed at local merge `218c5a2f86037bf786d6292c7dbb3673a326d457`. All fourteen
selected manuscript source inputs in that tree were compared byte-for-byte with
the named main commit. This comparison identified exactly four changed inputs.

The [machine-readable assessment](source-reassessment-20261116.json) records
before/after identities, inspected supplemental sources and the bounded claim
decisions. The [prior snapshot](source-reassessment-20261116.previous-snapshot.json)
is retained byte-for-byte; prior manuscript editions and render receipts retain
their own source identities.

## Changes assessed before updating fingerprints

| Changed source | Observed change | Claim decision |
| --- | --- | --- |
| `Cargo.toml` | Adds `innovate-kairos` to the workspace and excludes the standalone `innovate-games` crate. The workspace version and existing dependency declarations are unchanged. | Retain C01/C06's architecture and ownership statements. A workspace member is not accepted provider integration, distribution or research use. |
| `Cargo.lock` | Adds only the local `innovate-kairos` package, version `0.7.0-alpha.1`, depending on `innovate-core` and `serde_json`. No existing package identity, source or dependency record changes. | Retain C06's limited dependency-reuse statement. No fresh provider, security or installed-consumer qualification is inferred. |
| `rust/crates/innovate-core/Cargo.toml` | Adds the `workloads` benchmark target with `harness = false`; numerical dependencies are unchanged. | Retain C01/C06. A benchmark target is not a measured performance result; P03 remains withheld. |
| `rust/crates/innovate-tabular/src/lib.rs` | Appends Rust observation-panel binding, joint-draw IPC and cohort-occupancy IPC APIs. The previous file contents are preserved. Panel binding validates finite keyed observations but preserves input row order. | Retain C05's copying/LazyFrame and Julia/R limitations. Narrow C07's time-ordering example to documented model time grids rather than all input tables. |

The manuscript now says: “the documented model time grids require finite,
ordered values.” The unchanged native guide describes those model-call inputs;
the newly inspected typed `TimeGrid` constructor also requires finite, strictly
increasing values. This wording does not claim that panel ingestion sorts rows
or that every new IPC decoder has complete scientific validation. No new method,
performance, correctness or deployment claim was introduced.

## Claims and readiness retained

- C02's sixteen-family inventory and all 38 required migration obligations remain
  bound to the unchanged README, migration ledger and release gates. The version
  remains Python `0.7.0a1` / Cargo `0.7.0-alpha.1`.
- Existing Python/Julia/R binding implementations and the existing native request
  dispatcher did not change in this source comparison. New Rust table helpers
  do not establish completed Julia/R IPC adapters, native bundles or platform
  coverage.
- The empirical plan still leaves M01–M05, V05, Q02, Q03 and Q05 open. Its newly
  checked V01–V04/Q01/Q04 items identify source/protocol/test work; this assessment
  does not accept those checkboxes as executed empirical or release evidence.
- The new capability registry's four entries have null empirical and release
  receipts. Its implementation/test paths and application-promotion consumer
  names do not establish a version-bound research analysis, external adoption,
  accepted package, publication or reproducible empirical output.
- Authorship, affiliations, funding/conflicts, full AI history, research-use
  linkage, archival/registry outcomes and journal decisions remain pending.
  P01–P06 remain withheld.

The merge changes substantially more files than the fourteen selected manuscript
inputs. Supplemental domain, adapter, registry and empirical-plan sources were
inspected to interpret those changes. This is a bounded manuscript source
assessment, not a review or qualification of all new scientific routines.
Fingerprint equality remains an identity check rather than proof of truth.

## Edition and render evidence

The visible source revision in `paper.md`, the claim-register baseline and the
snapshot were updated only after the assessment above. The bibliography is
unchanged. The updated edition passed the offline integrity prerequisite
(fourteen source inputs, three edition inputs, six required sections and 1,076
prose words) and the actual Pandoc BibTeX/citeproc/official-TeX stages recorded in
[render attempt 3](joss-render-20261116/attempt-3/receipt.json).

The PDF attempt again stopped at the local environment's missing `luacode.sty`
(Make exit 2; wrapper exit 1). No PDF was produced or visually reviewed. This
reassessment installed no further TeX dependencies. Each render receipt
independently records its actual PDF and visual-review outcomes; earlier
source/build results do not transfer to a changed edition.

ChatGPT with GPT-6 Astra Pro assisted this source comparison, wording
clarification and evidence preparation. This records the assistance performed;
it supplies no human-author attestation or whole-project AI declaration.
