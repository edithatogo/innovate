# Implementation Plan

## Combined PR and main input reassessment — 22 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` reconciled PR #684 at
`5af412338d1cf46c502da89295f058816ca00e3f` with main commit
`a9360c1ea3414eaf0e712dbb131d3e923e958571`. Compared with the PR snapshot,
only the README and product input hashes changed in the combined tree. The
README adds the private security-reporting link. The product change records
Paignia and UOGTO responsibilities and preserves the requirement for provider
parity and acceptance. The main-branch reassessments below cover these two
changes; they do not add a scientific or completed-provider claim to the draft.

The combined snapshot retains the PR's ledger, lock, manuscript edition and
software baseline, together with the two reviewed main-branch input hashes.
All fourteen source and three edition hashes match the combined working tree.
The focused JOSS readiness suite passed on repo-local Python 3.15.2. The draft
checker passed and its submission-ready mode returned 1 as required. The
scientific, author, provider and release holds remain open.

## JOSS source review after adopter qualification

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` compared the 14 JOSS source
blobs at `c1399484aa7efa224304b0a84c559ecffc10726e` and
`dbb360d8e281ea103778484512e0a35958dfba83`. Only the migration ledger
changed. It records the qualified adopter classifier and 37 incomplete
obligations. The draft's count of 38 required obligations and its existing
scientific limitations still apply.

The 88 JOSS and release-claim tests passed on Python 3.15.2. The draft
checker validated the 14 source and three edition files. Its submission
check returned 1. The release inventory digest matches the three
verified wheel signatures.

Hosted job `135984167161` also found the registry-readiness packet's old
ledger digest. Its four input blobs were compared at the prior assessed
revision and `dbb360d8e281ea103778484512e0a35958dfba83`. The ledger was the
only changed input. The updated registry packet continues to list all three
destinations as pending. The registry and JOSS readiness suites passed 106
tests, and the registry checker reported a valid incomplete record.

## JOSS source review after the adopter classifier change

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the draft inputs and this
record in the PR #684 worktree
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`.
Hosted native job `135857338218` reported a stale lockfile fingerprint after
`c1399484aa7efa224304b0a84c559ecffc10726e`. The integrator checked all fourteen
input blobs against Git at both revisions and refreshed the source and edition
hashes in the draft. The core manifest and lock add a direct edge to the already
locked `num-bigint` 0.4.8.

Python 3.15.2 ran all 87 cases in the draft, setup and render-wrapper suites.
`scripts/check_joss_readiness.py --json` validated fourteen source hashes,
three edition hashes and six sections. `--require-submission-ready` returned
1. The pending render and prior receipts are recorded with their own editions.
The release source digest is
`ed085a5f9131fbd265e9bab54706fd6e0cf1811d3a7ecee1a08f4429a6c10eec`
with or without the draft update.

## JOSS source review after the calendar dependency change

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns this source review
in the PR #684 worktree
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate` from
`60c0593988f522495bbc867d91db3949c8086f4c`. Scope covers this
record, the JOSS source snapshot, working draft, claim register and README.
The draft checker reproduced a stale `Cargo.lock` fingerprint. Compare all
fourteen reviewed inputs with the prior source and retain the existing
unapproved status. The separate Inara workflow and its artifacts in PR #687
stay with their current owner.

The compared source inputs changed only in the core manifest and lock.
The snapshot identifies all fourteen Git inputs and three edition files.
The repo-local Python 3.15.2 draft and environment suites passed 69 cases.
Pinned Inara and Pandoc 3.6.4 built a fresh four-page draft. The integrator
viewed all four pages and checked the source revision, visible draft notice,
section order and references. The draft-template placeholder and sidebar
wrapping remain visible on page one. The rendered PDF SHA-256 was
`2c931607ac506ed09930f57a4610f02533f11d46f4f347eebae4ab055849c846`.
The draft remains unapproved and submission readiness remains false.

## JOSS setup diagnostics — 19 November 2026

Chat `01a1e448-3a28-7653-a681-cd3fdb41f498` owns the diagnostic correction
from `19603ac397a8e199eb5792559587fbf366f3df80` in
`/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`.
Scope covers the two configuration helpers flagged by the hosted scanner, their
reproduction instructions and `tests/native/test_joss_environment.py`.
Native scope is Innovate draft PR #684 and the hosted canary.

The two current scripts report stage and exit code. Their setup logs and
receipts remain in the caller's output directory. The reproduction instructions
now identify both original helper blobs and SHA-256 digests at
`9e12948a65d780e23f55dc343aa909e58257e569`. Both digests match the original
receipts. The commands for a new setup consistently select Python 3.15.2.

On the repo-local Python 3.15.2 interpreter,
`pytest tests/native/test_joss_environment.py -k logs_stay` reproduced the
console path disclosure in all four cases. The four corrected cases pass on
both success and failure. The existing JOSS environment, draft, registry and
readiness checks also pass. The next check is the hosted scan of the committed
source. This work supplies no new render or scientific acceptance evidence.

Hosted rescan `44478681210` covered all 174 non-Rust files and still
flagged the two subprocess return values going to the console. Project only
the stage and its pass or fail outcome. Retain the numeric codes in the
existing file receipt and preserve the actual process exit status.

## Product-boundary input reassessment — 22 November 2026

PR #700 head `54330fff28a81f790598361029bd5768a95a86a5` changes only one
of the fourteen JOSS source inputs: `conductor/product.md`. Its generic-game
paragraph now identifies Paignia as the reusable provider and UOGTO as the
semantic authority, with gradual migration after parity and consumer checks.
The other thirteen source inputs and all three manuscript-edition inputs match
their recorded fingerprints.

The affected source-supported claims C01 and C06 were reassessed: the Rust and
Python/Julia/R architecture is unchanged, as are the optional Kairos/Voiage
boundaries and their acceptance limits. The new game-ownership policy does not
claim completed provider adoption, scientific qualification or release. The
manuscript therefore needs no scientific prose change. Its recorded software
baseline, edition hashes and historical render receipts remain unchanged;
this is a reviewed product-document supplement to that baseline.

The previous product digest was
`20c3ed89c5afd4c097555dded6169ee1c01e0c6777b02ad2d0ffdab62b8def27`;
the reviewed input at the PR head is
`a69206d611af99b8a9e0c7d685f5fecd8b36570804b30ac63afd7908dba0f9e7`.
Only that entry in `joss/source-snapshot.json` changes. The checker has no
automatic hash-refresh mode and is unchanged. `submission_ready` and
`release_ready` remain false; all 38 migration obligations and the existing
author, rendering, scientific and provider acceptance holds remain in force.

Validation used repo-local CPython 3.15.2 and pytest 9.2.0. The pre-fix checker
reproduced `Input digest mismatch: conductor/product.md`. After the review:

- `python -m pytest tests/native/test_joss_readiness.py`: 50 passed.
- `python scripts/check_joss_readiness.py --json`: exit 0; fourteen source and
  three edition inputs verified, submission readiness false.
- The same checker with `--require-submission-ready`: exit 1 as required.
- `python scripts/check_migration.py --release`: exit 1 with all 38 existing
  obligations still unqualified.
- `git diff --check`: passed.

No native, release or hosted pass is inferred from this input repair.

## Separate evidence acquisition — 16 November 2026

Starting PR head `d36c33f6b4fb480828ddb9aae0121f3379667015`; assessed main
`b29a1d27f54fdd08d77881ee999772fceab7c03a`, consumed in the isolated integration
commit `2748084292e5e57c181c1665a686ad3a291145bf`.

- [x] JOSS-02f: review the changed source input, complete the official draft PDF
  in an isolated, pinned TeX dependency closure, and visually inspect the result.
  Final attempt4 is a four-page official draft; compilation and all-page visual
  review pass, with the template placeholders/sidebar caveat explicitly retained.
- [x] JOSS-02g: protect the Mesa author family name `ter Hoeven` in the
  BibTeX after actual PDF review found its particle dropped in citations. This
  narrow publisher-supported correction extends the allowed edition scope to
  `joss/paper.bib` and its reviewed fingerprint; preserve attempt 3 and create
  a fresh render, without changing scientific prose or author attestations.
- [x] JOSS-02h: harden the reproducible environment helpers after independent
  review reproduced unsafe reuse of an existing setup root. Require a pinned
  database before writes, clean copied-font sources and absolute owned TeX
  executables; preserve original executed helper bytes and verify the corrected
  setup in fresh directories. Scope includes `tests/native/test_joss_environment.py`.
  All 15 guard tests and an actual cached replay pass; 241 archives, 24 fonts
  and owned binaries match the original environment. The final six-suite run
  records 208 passes/no skips. Original PDF provenance remains separate.
- [x] JOSS-01b: acquire primary public distribution/development history and
  investigate identifiable version-bound research use without treating teaching
  examples or unexecuted downstream plans as research use.
- [x] SWH-01b / RRID-01b: follow public provider identifiers and distinguish
  package archival evidence from the selected release and canonical resource.
  Existing PyPI 0.5.0/0.6.0 archives and resolving SWHIDs are verified;
  downloaded sdists match their recorded hashes and canonical metadata.
  Actual research use and an Innovate RRID remain unverified.
- [x] REG-16c: reconcile #668/main and the new #671 documentation ownership,
  update actual readiness findings and prepare a concrete integration handoff.

Allowed edits remain the two owning track namespaces and separately scoped
scientific tooling. All prior attempts and receipts are preserved. External
correspondence, submission, registration, release and main promotion remain
separate from this evidence acquisition.

## Reassessment after the main advance — 16 November 2026

Main is now `57d2916586d107c8affe90355e453abe27f01c26` after the external
merge of #656. The [integrated-source record](integration-20261116.md) supersedes
earlier current-main and open-PR wording below while preserving each receipt.
Allowed edits remain within this track and the scientific qualification track;
no shared source, pin, workflow, registry or other writer's environment changes.

- [x] REG-16b: consume the merged source in an isolated worktree, verify all
  four registry input fingerprints against both Git revisions and local bytes,
  and retain pending provider states. The former three Conductor registry
  discrepancies are resolved by the upstream merge; local hygiene now passes.
- [x] JOSS-02d: review the four changed manuscript inputs and each affected
  claim, retain the prior snapshot, and explicitly bind the new unapproved
  edition to main. Clarify model time-grid ordering without extending that
  claim to every newly added table API; preserve scientific and author limits.
  The new 1,076-word edition passes all 14 source and three edition identities.
- [x] JOSS-02e: rerun pinned bibliography/TeX processing against the new edition.
  Attempt 3 passes real bibliography, official TeX and draft-marker checks;
  PDF still fails on missing `luacode.sty` (Make 2, wrapper 1), with no PDF or
  visual-review pass. The prior edition and attempts remain preserved.
- [x] Repeat all four preparation suites against the reassessed integrated
  tree: **169 passed, zero skips**. Explicit submission mode still exits 1.
- [ ] Integration: reconcile current hosted checks and hand off exact failures
  and remaining shared work to #624. Native API documentation, recurring CI,
  scientific/research-use evidence, author decisions and external outcomes
  remain separate gates.

## Executable preparation follow-up — 16 November 2026

The user's follow-up authorises implementing the remaining feasible preparation.
Starting published head: `3f7e39249de496dc28a59e9e94a3c01390d1cb4b`
(local tree-equivalent receipt `0c718cea3eb982717ce940560868004d71261e9b`).
Inspected main: `18c467e18b71d972060365c68865aa405a69949c`; its only advance
from the assessed source changes the Renovate workflow. #624 still owns shared
integration; #656/#659 have no overlap with the allowed files below.

- [x] JOSS-02a: implement an offline manuscript/source integrity check with
  negative tests for drift, incomplete evidence and false readiness; preserve
  the current unapproved edition and all scientific/author/provider gates.
  Fifty tests and independent review pass; 14 source hashes also match Git blobs
  at the recorded assessment revision. The checker itself does not claim Git
  authentication or submission readiness.
- [x] JOSS-02b: implement the reproducible pinned draft-render wrapper and its
  18 guard tests; execute actual bibliography parsing, official TeX/citeproc and
  draft-marker checks. The final `attempt-2` receipt matches the final script.
- [!] JOSS-02c: complete PDF compilation and visual inspection. Actual LuaLaTeX
  stopped at missing `luacode.sty`; Make exited 2, wrapper exited 1 and no PDF
  was produced. Continue in a complete TeX/Inara environment with a new receipt;
  do not infer visual acceptance from TeX output.
- [x] SWH-01a: resolve the historical release/tag inventory, including the
  annotated-tag target; preserve target selection, rights and archival gates.
- [x] Reconcile observed hosted results and remaining JOSS-01, RRID-01, SWH-01,
  author and external gates after execution; independently review the bounded
  implementation and final portable fixtures. The exact publication, issue
  readback and new-head hosted receipts belong in #662 and its #624 handoff.

Allowed files: this track, `scripts/check_joss_readiness.py`,
`tests/native/test_joss_readiness.py`, `tests/native/test_joss_render.py`, and
the separately scoped scientific Arrow qualification packet. Root workflows,
manifests, dependency pins, common registries, empirical protocols and other
owners' source remain reserved.
Preparation checks cannot supply research-use facts or author attestations.

## Scientific and journal reconciliation — 16 November 2026

Current work packet: `REG-16`, based on
`69b674814513e09a683279c679b1584af8bb4214`. This section and
[the current reconciliation](reconciliation-20261116.md) supersede operational
status in the dated receipts below; those receipts remain historical evidence.
Allowed changes are this track and the existing scientific ecosystem alignment
track. The shared integrator in #624 retains promotion, root registry, workflows,
pins and scientific acceptance. #656 owns the overlapping empirical plan and
#659 retains documentation work.

- [x] Reconcile the five requested issues, merged/closed PRs, current metadata,
  scientific dependencies and the actual native issue hierarchy.
- [x] Prepare a separate JOSS-format draft, checked bibliography and claim/source
  register; preserve the legacy manuscript and V05-owned scholarly corrections.
- [x] Refresh bounded read-only provider lookups and distinguish absent tested
  origins, inaccessible results, submission and verified external outcomes.
- [x] Verify the updated pending record, manuscript structure/citations/links,
  exact changed-file scope and independent review; retain real failing gates.
- [x] Publish the bounded PR and reconcile all five issue descriptions with its
  exact source/evidence. Published as draft [#662](https://github.com/edithatogo/innovate/pull/662),
  initial head `0b5751002c7fd6cfe6ebd60b16208c94597cfad2`; all five issue bodies
  were updated and read back on 16 November 2026. Integration and external
  completion remain separate.

The remaining executable packages and their completion evidence are in the
[current reconciliation](reconciliation-20261116.md#ordered-work-and-acceptance).
No new scientific implementation owner is assigned by this packet.
## Packaging-input reassessment — alert 44, 16 November 2026

The alert 44 candidate based on `b29a1d27f54fdd08d77881ee999772fceab7c03a`
changes the Python manifest's licence-file patterns and source-distribution
include list to retain the vendored upstream licences, notices and sources.
The project name, development version, description, Python constraint, declared
project licence, authors, runtime dependency and repository identity are
unchanged. A wheel built from an extracted source distribution contains the
project licence and the copied upstream licence and notice files. The dossier's
distinction between project-owned Apache-2.0 code and applicable third-party
terms remains necessary; this check does not establish publication rights.

The previous manifest digest was
`888cfd6e4611a6e8c526db96e34b0d85167fb20f3dea6fb37b537fe6dd6ee395`;
the reassessed candidate digest is
`d0269f89303986a3cd40bde7c98c5ed45e96b21cce0d84bb3d9e1b556041936c`.
The original assessed revision remains the upstream reconciliation baseline.
Software Heritage, RRID and JOSS remain pending, and the release and scientific
gates remain open. No external lookup, submission or acceptance is inferred
from this packaging reassessment.

The subsequent wheel-content check found identical upstream licence texts in
separate wheel paths. The manifest now includes the project licence and a
generated third-party bundle that maps each upstream path to its unmodified
notice text, grouping only byte-identical copies. Individual upstream licence
files remain in the source distribution. The reassessed manifest digest is
now `9f6cdc01a00664e5cae53192f261891d3b440aa68a26ed5173ba8fe3a4110a05`;
identity, version, project licence and the three pending registry dispositions
are unchanged. This metadata review does not establish publication rights.

## Phase 1: Readiness and prerequisites

- [x] Inspect the current native repository and registry-specific prerequisites.
  Base: `276c2e505a0ce26910820fe8a94a6a498f505700`; native development version
  `0.7.0a1`, Apache-2.0, Rust/Polars with Python/Julia/R bindings. The release
  gate remains closed; metadata presence is not release or rights acceptance.
- [x] Reconcile GitHub ownership and hierarchy. Read-only REST inspection of
  `/repos/edithatogo/innovate/issues/398/sub_issues` confirmed native edges to
  open #399, #400 and #401. PR #402 is closed/unmerged. Concurrent PRs #590,
  #600, #602 and #603 own separate security/native/quality work.
- [x] Capture current repository-specific checks and their actual outcomes below.
  Earlier July completion markers are historical, not native-release evidence.

## Phase 2: Registry deliverables

- [x] Correct development citation metadata and the placeholder author ORCID;
  retain the existing author affiliation pending publication-specific confirmation.
- [x] Implement the current source-fingerprinted readiness record and offline
  integrity checker; move behavioural coverage into the active `tests/native`
  suite and replace the old string-only readiness test.
- [x] Prepare Software Heritage/RRID metadata, normalized read-only lookup
  receipts, official eligibility sources, and a JOSS assessment/content outline.
- [x] Replace the unsupported repository-ready claim with explicit preparation
  and external gates; expose the contract in the active documentation site.
- [ ] [Issue #399](https://github.com/edithatogo/innovate/issues/399): selected
  release, successful archival and resolving matching SWHID remain pending.
- [ ] [Issue #400](https://github.com/edithatogo/innovate/issues/400): duplicate
  check, submission, curator decision and resolving matching RRID remain pending.
- [ ] [Issue #401](https://github.com/edithatogo/innovate/issues/401): current
  qualified manuscript/software/research-use evidence and author declarations
  remain prerequisites before submission and editorial/publication evidence.

## Phase 3: Reconciliation and closeout

- [x] Run independent source review and apply the two identified fixes: reject
  query/fragment identifiers as resolver URLs, and reject negative SWH results
  while accepting the provider's structured successful resolver response.
- [x] Reconcile track metadata, current documentation and native issue hierarchy.
- [ ] Verify GitHub project placement with an accessible project-membership
  surface; this increment does not claim the project-visible acceptance criterion.
- [ ] Obtain final-head hosted CI evidence and the existing integrator's merge
  window. Do not race concurrent promotion or weaken required checks.
- [ ] Close registry deliverables only with authoritative provider evidence or
  an evidenced ineligibility decision; keep parent #398 open while those remain.
- [ ] Archive this track only after all automatable work and external gates have
  been reconciled. This increment does not complete the whole track.

## Verification receipt — 14 November 2026

Owned interpreter: `.venv-registry/bin/python`, observed CPython **3.15.2**.
Focused tools: pytest **9.2.0**, pytest-cov **7.1.0**, Ruff **0.17.4**,
coverage **7.17.3**, cffconvert **2.0.0**. This isolated environment does not
change the repository lockfile or another worker's environment.

| Command / check | Observed outcome |
| --- | --- |
| Initial `python -m pytest tests/native/test_registry_readiness.py --maxfail=1` before implementation | Failed because the new checker module did not yet exist. This is availability evidence, not a behavioural or mathematical regression. |
| `.venv-registry/bin/python -m pytest tests/native/test_registry_readiness.py` | **56 passed**. Independent reviewer repeated the same suite successfully and reproduced/rechecked both review findings. |
| Focused coverage with branch measurement, restricted to `scripts/check_registry_readiness.py` | **138/144 statements (95.8%)**, **24/30 branches (80.0%)**; 93.1% combined. Coverage belongs only to the metadata checker; it is not native/scientific coverage. The initial module-name selector collected no data and is not counted. |
| `.venv-registry/bin/ruff check scripts/check_registry_readiness.py tests/native/test_registry_readiness.py` | Passed after formatting. |
| `.venv-registry/bin/ruff format --check scripts/check_registry_readiness.py tests/native/test_registry_readiness.py` | Passed. |
| `.venv-registry/bin/cffconvert --validate -i CITATION.cff` | Valid against CFF **1.2.0**. |
| `.venv-registry/bin/python scripts/check_registry_readiness.py --json` | Exit **0**; record valid, recorded_complete false, online_verification_performed false; #399/#400/#401 pending. |
| Same checker with `--require-complete --json` | Exit **1**, intentionally preserving all three unresolved destinations. |
| `.venv-registry/bin/python scripts/conductor_registry_hygiene.py --json` | Passed; no missing links, unregistered/orphaned tracks, or wrong targets. |
| `.venv-registry/bin/python scripts/sync_versions.py --check` | Native development mapping valid; release remains gated. |
| `.venv-registry/bin/python scripts/check_migration.py` | Development policy valid; **38** scientific obligations remain. |
| Same migration checker with `--release` | Exit **1**, expected: scientific migration and acceptance are incomplete. |
| Local Markdown-link validation and `git diff --check` | Passed. |

The receipt and source revision for hosted validation belong in the PR. No full
native suite, installed-package acceptance, docs build, provider submission,
publication, author attestation, or external acceptance is inferred from these
local checks. Local Node is 24.22.1; the documentation build requires Node 26,
so the existing qualified hosted documentation workflow must supply that gate.

## Community navigation reassessment — 22 November 2026

The Community Standards increment appends a Security section to the README,
linking the existing private vulnerability-reporting policy. It changes no
software capability, scientific statement, usage example, licence term or
manuscript claim. The existing JOSS packet was reassessed for that bounded
packaging-input change; only the README fingerprint was updated from
`35c86f3d89d6187cffe62a37ccb67d061f530ec2616768c5aae008ba2cf309be` to
`9aba1db7cec72e9b47f4ebc9b3a788e80c2f3c9094b45cb55ab14d4869e19cc4`.
The other thirteen source fingerprints, three edition fingerprints and
original software revision remain unchanged from the prior reassessment.

Under the existing CPython 3.15.2 repair environment,
`python -m pytest tests/native/test_joss_readiness.py -q` passed all 50 cases.
`python scripts/check_joss_readiness.py --json` exited 0 with all fourteen
source and three edition inputs valid. It retains `submission_ready: false`,
`scientific_qualification_performed: false` and `git_revision_verified: false`.
No provider submission, scientific release hold or author declaration changes.
