# Reassessment after the frontier merge — 16 November 2026

This record continues [the implemented preparation](implementation-20261116.md)
after an external main-branch advance. It governs the current source assessment;
earlier receipts remain evidence for their original sources and artifacts.

## Source and allowed work

PR [#656](https://github.com/edithatogo/innovate/pull/656), head
`ca9739d59d7467180ddce2eb5c411021ddb706f9`, was merged by another actor at
11:10:35 UTC as `57d2916586d107c8affe90355e453abe27f01c26`. This followed
the Renovate-only commit `18c467e18b71d972060365c68865aa405a69949c`.
The combined difference from the initial assessment at
`69b674814513e09a683279c679b1584af8bb4214` is 227 paths.

Our published implementation head `a49d453e8786eca78ad89055a442862b3afcf9d2`
was tested by GitHub both directly and as synthetic merge
`b4765edda6def0cbfa16722f73d0981383893cca`. The synthetic merge's parents
are the new main and that PR head; the PR metadata's older base SHA did not
describe this executed checkout. The isolated local integration commit
`218c5a2f86037bf786d6292c7dbb3673a326d457` has the same complete tree,
`81dffc475d082208cb5563747b685b4451cd6397`.

This continuation consumes the already-merged main source and changes only the
two scientific/journal track namespaces. It reassesses the draft and records
fresh affected execution. It changes no scientific implementation, manifest,
lock, provider pin, shared workflow, common registry or quality threshold.
The sole #624 integrator retains promotion and shared work. Open documentation
PR #659 remains at `178779bd21d7fdde2fe4f6b8179444b72a4bc6d1`; its owner
is preserved.

## Manuscript and registry reassessment

The integrity command correctly rejected the combined tree before any update:
`Input digest mismatch: Cargo.toml`, exit 1. Four of the fourteen assessed
source files changed. These changes require claim review and new edition
fingerprints; a new Git revision alone cannot supply that review.

| Changed input | Change relevant to the draft | Required disposition |
| --- | --- | --- |
| `Cargo.toml` | Adds `innovate-kairos` as a fifth workspace crate and explicitly excludes the standalone games crate | Keep workspace and distribution scope explicit; do not infer acceptance of either adapter |
| `Cargo.lock` | Adds the local `innovate-kairos` package | Record the changed build input; retain the existing provider and dependency identities |
| `rust/crates/innovate-core/Cargo.toml` | Adds the native workload benchmark target | Do not infer measured performance or comparative advantage from a benchmark declaration |
| `rust/crates/innovate-tabular/src/lib.rs` | Appends panel, joint-draw and cohort APIs while retaining the existing scenario IPC code | Recheck the table and validation claims; avoid a blanket ordering claim covering every new API |

The new edition retains bounded architecture, transport and research-use
limitations. Newly merged implementations, archived plan statuses and capability
entries are not independent evidence of scientific acceptance, research use or
release readiness. In particular, the migration ledger still contains all
38 unresolved obligations.

The [claim-by-claim review](evidence/source-reassessment-20261116.md) and its
machine-readable companion retain the prior snapshot and the reasons for each
decision. The current draft has **1,076 prose words**. All fourteen source and
three edition identities pass the integrity command; its explicit submission
mode still exits 1. All **169 preparation guards pass again**, with zero skips,
on this reassessed integrated tree. The three new tool implementations and
their tests are unchanged from the reviewed first implementation.
The [final preparation checks](evidence/integrated-preparation-checks-20261116.json)
retain the executed result, source comparisons, unchanged tooling hashes and
independent review. Authored Markdown/JSON/Python/XML whitespace checks pass.
The full staged diff check reports whitespace in four verbatim generated log/TeX
files; their hashed output bytes are preserved. Earlier unstaged checks did not
include those then-untracked files, so they are not a clean full-evidence check.

[Render attempt 3](evidence/joss-render-20261116/attempt-3/receipt.json) executes
the actual pinned pipeline for the changed edition: bibliography parsing,
official TeX/citeproc and draft markers pass. LuaLaTeX again stops at missing
`luacode.sty` (Make exit 2, wrapper exit 1); no PDF or visual-review pass exists.
No additional TeX installation was attempted in this reassessment.

The four registry inputs (`CITATION.cff`, `LICENSE`, `pyproject.toml` and
`migration-ledger.json`) are unchanged from the prior assessment. Their existing
fingerprints were verified against Git blobs at both assessed revisions and the
local files before moving `readiness.json`'s assessed revision. The
[repository-check receipt](evidence/integration-repository-checks-20261116.json)
retains those comparisons and executed commands. Provider states remain pending
and no external lookup or outcome is inferred from this source update.

## Affected native execution

The earlier [122-input execution packet](../scientific_ecosystem_alignment_20261029/evidence/arrow-installed-20261116/README.md)
is preserved. The integrated source contains 175 tracked build inputs, so the
earlier wheel and minimum-compiler result cannot be promoted to the new source.
The new [integrated execution packet](../scientific_ecosystem_alignment_20261029/evidence/arrow-integrated-20261116/README.md)
records the fresh wheel, installed Arrow selection, native targets and separate
minimum-compiler check with their actual outcomes.

| Integrated check | Actual outcome |
| --- | --- |
| Fresh installed development wheel | **42 passed, zero skips/failures/errors**, including all four optional PyArrow cases; all 175 source inputs unchanged |
| Native scenario exports, Rust 1.100.0 | **8 passed**, zero failed/ignored/filtered |
| Existing provider/network targets, Rust 1.100.0 | **64 passed** (17/37/5/5), zero failed/ignored/filtered |
| Actual Rust 1.99.1, locked workspace/all-target check | **Passed** for all five workspace crates on Linux/default features; compile checking only |

The rebuilt wheel has SHA-256
`af3fad4f104f3b8ffad6ba92fe139f9c8a7183ee85e0a9ecd938f6e2dc015d3c`.
Independent review verified the source hashes against current bytes and the
recorded integration commit, and checked wheel/installed/native/RECORD, JUnit
selection and log identities. The final documentation commit may differ from
the build commit; unchanged native input hashes establish only the stated
source-input equivalence for this artifact.

The standalone `innovate-games` crate is excluded from the Cargo workspace.
Capturing its files for source identity does not qualify that crate through a
workspace build. Platform, optimized artifact, provider publication and the
38 scientific obligations retain their separate requirements.

## Current repository and hosted findings

On the integrated source, local repository-policy and development-version checks
pass. Conductor registry hygiene now passes with seven active directories and
no missing, orphaned, stale or wrong-target entries: #656 resolved the three
previously recorded archive/status discrepancies. That structural result does
not ratify scientific completion claims in archived tracks.

The inspected hosted runs on the earlier PR head distinguish their checkouts:

| Actual checkout and job | Observed result | Interpretation |
| --- | --- | --- |
| `a49d453…`, Python coverage job [133258448405](https://github.com/edithatogo/innovate/actions/runs/43855460409/job/133258448405) | 480 passed, 5 skipped; binding line coverage 86% | Direct PR-head result; the four optional Arrow skips are separately addressed by the installed qualifier, and the live OECD skip is retained |
| `b4765edd…`, release job [133258448506](https://github.com/edithatogo/innovate/actions/runs/43855460460/job/133258448506) | Exit 1 at migration policy, listing the same 38 obligations | The release gate remains intact on the combined source |
| `b4765edd…`, native documentation job [133258448304](https://github.com/edithatogo/innovate/actions/runs/43855460465/job/133258448304) | Exit 101; 176 missing-documentation diagnostics in the newly merged Rust surface | A concrete integrated-source blocker for #624 and the existing source/documentation owners; no threshold relaxation or competing source edit here |
| `b4765edd…`, Ubuntu native job [133258448385](https://github.com/edithatogo/innovate/actions/runs/43855460477/job/133258448385) | Rust: 457 passed, 1 skipped. Python: 26 failed, 461 passed, 5 skipped; all failures are manuscript-integrity cases encountering `Cargo.toml` drift | This packet reviews and updates the affected edition, then passes the 169 preparation guards locally; final-head hosted execution is still required |

The [hosted-blocker receipt](evidence/hosted-integration-blockers-20261116.json)
records the release and documentation gates and the sixteen affected Rust source paths. None
of those paths appears in the inspected #659 diff; #624 must reconcile the
remediation assignment with the existing source/documentation owners.

The final published head, full changed-file boundary, issue readbacks and hosted
results belong in the subsequent #662 receipt and #624 handoff. These earlier
jobs do not establish a pass for a later head. Queued offline-security work and
unexecuted scheduled consumers remain unqualified.

## Remaining decisions and external evidence

Recurring hosted optional-Arrow and minimum-compiler qualification remain
shared-CI intake. Native public-documentation failures must be repaired by the
existing owners and the resulting exact source rechecked. The current draft's
PDF/visual result, author declarations, actual version-bound research use,
public-history assessment, archival target/rights and provider outcomes remain
governed by the track plan and readiness record.

This continuation performs no main merge, release, provider registration,
archival request, journal correspondence or submission. The local integration
commit is a review and execution workspace, not a promotion decision.
