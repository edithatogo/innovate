# Implementation plan

## Reconciliation

- [x] Inspect live main, open PR ownership, existing Astro site, archived tracks and published site.
- [x] Identify removed-API quickstart, missing documentation checks and the explicit deployment gate.

## Executable contract increment

- [x] Add fail-closed Python docstring discovery, strict completeness and bounded regression contracts with adversarial tests.
- [x] Add page/example discovery, tutorial requirements and exact-source execution evidence with adversarial tests.
- [x] Complete maintained Python, Julia and R binding documentation without changing runtime behavior.
- [x] Complete and compiler-check Rust public documentation; check actual PyO3 exports against stubs and runtime docstrings.
- [x] Replace the quickstart with an imported, asserted native Polars example and accurate interpretation.
- [x] Qualify the scenario tutorial and all eight current native Python reference examples; remove every accepted content exception.
- [x] Generate source-bound current Python, Julia and R references into the actual Astro collection.
- [x] Qualify the final Astro artifact and restore build/type-check/contract workflow steps.
- [~] Gate publication on executed Julia/R documentation examples and source-bound language receipts, preserving honest local versus hosted results.
- [x] Run independent review, apply findings and record exact commands and limitations.
- [x] Integrate the initial contracts through PR #610 and the aggregate receipt repair through PR #611; preserve coordinator merge ownership.
- [~] Qualify follow-up PR #613 at its final exact revision, including browser behavior and migrated content.

## Full-quality acceptance remains open

- [x] Reach zero structural debt for current pages and execute all ten current Python examples without accepted exceptions.
- [x] Complete compiler-verified Rust public documentation and executable rustdoc examples.
- [ ] Qualify every claimed supported language tutorial against its installed artifact.
- [~] Verify browser accessibility, keyboard/search/mobile behavior and deterministic resource budgets on the built artifact.
- [~] Review scientific claims against primary references and record worked reader tasks; human reader validation remains open.
- [ ] Verify the qualified Astro artifact at the public URL after authorized deployment activation and integration.

## Follow-up implementation — 15 November 2026

- [x] Reconcile receiving main `384ed3b0f42d967f520dc66ef536d8fa0d92df7d`, including merged PRs #610 and #611.
- [x] Recover twelve migrated current pages and their exact source bindings after the local executor disconnected.
- [x] Correct the entry/tutorial explanations and retain a scoped primary-source review with worked learner answers.
- [x] Promote current guides/tutorials in navigation and map all remaining historical current-path routes; preserve the versioned archive.
- [x] Require bundled Pagefind exclusion for historical current-path pages and retain searchability for current pages.
- [x] Add a source-freshness and claim-consistency gate for the scoped scientific review record.
- [~] Run complete contract-derived Chromium desktop/mobile qualification with axe, keyboard/search/version checks, overflow and resource budgets.
- [~] Qualify the narrow Julia world-age repair with `--depwarn=error` and fresh native receipts.
- [x] Recover the exact original browser dependency lock after workspace restoration, remove the unused temporary recovery workflow, and pass frozen installation with the existing supply-chain policy.
- [ ] Verify the final exact revision through the hosted build, browser and native documentation jobs.
- [ ] Verify the approved Astro publication at the public URL through the existing integration and Pages activation process.

PR #610 merged at `f6c7d058466ae5fe15b9846769cc0e6ba6774847`; PR #611
merged at `384ed3b0f42d967f520dc66ef536d8fa0d92df7d`.
The follow-up is [PR #613](https://github.com/edithatogo/innovate/pull/613).

The local executor failed during this follow-up. Recovered edits were read back
from immutable Git objects, but this recovery is not runtime or build evidence.
The original results below remain attributed to their original source. See
[native-runtime-followup.md](native-runtime-followup.md) for verified historical
native artifacts and the separately tested aggregate repair.

## Follow-up local qualification

The integrated follow-up passes 233 Python contract tests on Python 3.15.2, all
ten Node tests, frozen dependency installation, the 202-page Astro build, type
checking, source-bound content checks, and production artifact verification.
All ten Python examples and eighteen API doctest statements pass. The local
full-language command correctly remains nonzero without fresh Julia/R receipts.

The hosted #611 documentation job exposed a second aggregate hash-format defect:
the inventory's example digests are bare SHA-256, whereas its source map labels
those same digests with `sha256:`. The follow-up fixes only that comparison,
preserves exact receipt/source-map equality, and adds five complete-receipt
regressions that reproduce the Julia and R failure before the fix.

Browser discovery enumerates 142 cases. Actual browser execution, fresh hosted
native qualification, human reader trials and live publication remain separate.
Only superseded PR documentation runs may now be cancelled; main publication
retains its ordered queue, both qualification dependencies and explicit activation.

See [followup-qualification.md](followup-qualification.md) and
[the source-fingerprinted local record](followup-local-evidence.json).

## Observed browser failures and repair

The first full hosted Chromium run executed all 142 cases: 61 passed and 81
failed. It exposed accessible-name, light-theme contrast and scrollable-table
defects, plus a search-test assumption about Pagefind's Escape reset.
The bounded repair preserves the complete suite and thresholds, and adds actual
arrow-key scrolling checks within the existing page cases. The repaired Astro
build, type check, Node tests, review gate and artifact verifier pass locally.
Fresh hosted browser and native results remain required.

The receiving main has advanced to 663002b03ccb92abee2ece934d69116cb4bf3415
through merged PR #612. Its sole change is the accepted Renovate action update.
See [the exact browser failure and repair record](browser-repair.md).

## Evidence

Initial source: `13bdcf10ccb3e1cd06da1df4d98fc5da831768db`.
No historical `passed` status is reused as evidence for this increment.

Local qualification uses Python 3.15.2, Rust 1.99.1, Node 26.13.1 and pnpm
12.18.0. The locked native package was built from this checkout. The scientific
implementation is unchanged: Python ASTs, Julia source without docstrings, R
tokens without comments, and Rust source without documentation comments match
the starting revision.

Verified locally on the complete proposed source tree:

- Python source inventory: 55 documented symbols, including 40 public symbols,
  across two maintained source/stub files; strict structural gate passes.
- Installed native extension: all ten public callables match the stub and have
  substantive runtime argument/return/error documentation.
- Ten Python example files execute in the installed native environment with
  observed assertions and exact expected stdout. Eighteen Python doctest
  statements pass against the same source.
- `RUSTDOCFLAGS='-D missing_docs -D rustdoc::broken_intra_doc_links -D rustdoc::invalid_codeblock_attributes' cargo doc --workspace --no-deps --locked -j 1`
  passes for all four workspace crates. `cargo test --workspace --doc --locked
  -j 1` passes both existing examples; zero-test crates are not counted as
  additional executed examples.
- Julia documentation covers 24 exports (23 functions and one exception); R
  covers 24 functions across 13 function manuals plus one package manual.
  Structural checks pass. The exact inventories require seven Julia examples
  with nine observed assertions and six R examples with ten observed assertions.
  Julia and R interpreters are unavailable locally; their native execution is a
  hosted qualification obligation. The actual combined local execution command
  exits nonzero for these two missing receipts while retaining passing Python
  results; it does not certify the complete three-language scope.
- API generation and its read-only freshness check pass: 34 pages cover all
  declared exports (Python 40 symbols across 32 pages, Julia 24 across one page,
  R 24 across one page).
- `pnpm build` produces 199 HTML pages. `pnpm check` reports zero errors,
  warnings or hints. The production verifier checks 13,530 local references
  with zero errors, along with canonical routes, sitemap/search assets,
  document metadata and the configured per-page size budget. Its 305 external
  URLs are explicitly unverified.
- The current content inventory has 45 pages, zero unbound Python code blocks,
  zero structural violations and zero accepted baseline exceptions. Its 73
  historical pages and 80 versioned archive pages remain unqualified and outside
  current-content completion.
- All 198 Python contract tests and both Node content-transform tests pass,
  with no skipped tests. Measured coverage across the six Python checkers is
  85.59% of lines and 80.24% of branches (83.85% combined).
- Scoped Ruff checks and formatting pass. `actionlint` passes both changed
  workflows; `zizmor` reports no findings in offline mode. The release C ABI
  build also passes with the locked Rust toolchain.
- Independent review closed comment-hidden display bindings, stale/duplicate
  baseline exceptions, source/runtime export mismatch and execution-free
  completion. Actual doctest execution also corrected the mistaken assumption
  that every successful envelope includes an `error` key.
- The dark theme's accent text used a low-contrast light-theme palette. Separate
  theme tokens change the relevant calculated text/background contrast from
  1.75:1 to 13.37:1 in dark mode; light-mode link contrast is 5.14:1. These pairwise
  calculations are not a complete accessibility audit.
- Conductor registry hygiene passes: 106 entries, 13 active directories, no
  missing/stale/orphan links or directories.

The browser connection did not complete and local preview navigation was aborted
before dispatch. No browser screenshot, keyboard/mobile/search interaction or
assistive-technology result is claimed. The published URL still requires the
existing integration and explicit Pages activation process. The 153 retained
historical pages and all scientific release obligations remain separately open.

## PR 610 integration repair — 15 November 2026

- [x] Diagnose the exact-head documentation artifact failure from hosted logs.
- [x] Add six dependency-free workflow regressions; reproduce the defect before
  the change and verify the narrow repair locally on CPython 3.13.5.
- [x] Explicitly permit hidden receipt uploads using per-job filename allowlists;
  preserve empty-artifact failure and both publication qualification dependencies.
- [ ] Qualify the repaired head on hosted Python 3.15.2, including actual uploads
  and the previously skipped native documentation examples.

See [the bounded repair evidence](evidence-upload-repair.md) for exact source
identity, failing job, commands, local-runtime limitations and preserved gates.
The historical qualification above is not reattributed to the repaired head.

## PR 616 documentation example receipts

On `c24638b4ac4a0948dc2853c2ebd079ee6e95393e`, the aggregate check
rejected the Julia and R receipts because the tracked source map prefixes
digests with `sha256:` and example records use bare digests. The same check
failed on main `663002b03ccb92abee2ece934d69116cb4bf3415`.
[PR 616 job](https://github.com/edithatogo/innovate/actions/runs/43659146178/job/132645986240)
[Main run](https://github.com/edithatogo/innovate/actions/runs/43652562275)

This repair brings forward the hash comparison and regressions from [PR 613](https://github.com/edithatogo/innovate/pull/613).
On Python 3.15.2, the new case reproduced the failure for both languages.
After the repair, 14 source-hash, 60 content, and six documentation workflow
tests passed using `.venv-native/bin/python -m unittest discover -s tests -p`
with `test_documentation_source_hash_contract.py`, `test_documentation_content_contract.py`,
and `test_documentation_workflow.py`. Fresh hosted native receipts are required
to qualify the repaired PR.

## PR 656 documentation and coverage repair — 16 November 2026

- [~] Restore the existing strict Rust documentation check and add independent
  native tests for uncovered changed behavior; retain all coverage thresholds.
- Ownership: chat `01a1e3e9-537d-72a0-acc9-56b9ba954680`, worktree
  `/Users/doughnut/.codex/worktrees/innovate-pr656-doc-coverage/innovate`, branch
  `codex/pr656-doc-coverage`, base `8244d1eb23fec915c0ce93f4da93e3cae44d6ddb`.
  Initial allowed paths are public-item documentation in `innovate-core` source
  files outside PR #665's active paths, new focused `innovate-core` integration
  tests, and this plan. PR #665 retains its existing files until its handoff is
  integrated or ownership is otherwise reconciled. No tenant object or native
  authentication scope is assigned to this lane.
- Next operation: use the retained exact-head Rust coverage report to select
  behavioral and negative tests, and document public items with semantics
  supported by their implementation. Keep scientific-readiness, offline-security
  and coverage configuration unchanged. Hosted acceptance remains pending.
- PR #665 was integrated as `ca9739d59d7467180ddce2eb5c411021ddb706f9`;
  this owned branch advanced to that commit with the plan edit preserved.
  The documentation scope now includes the previously overlapping public items.
  The retained `8244d1eb` coverage report is a targeting baseline, not a
  coverage result for the advanced source.
- The first strict workspace Rustdoc run cleared the reported core errors and
  exposed undocumented public fields in `innovate-tabular` and `innovate-kairos`.
  This lane also owns documentation comments on those two source files; their
  interfaces and runtime behavior remain unchanged.
- Local qualification on macOS: Rust/Cargo 1.100.0, `cargo-llvm-cov 0.9.1`
  (official release asset SHA-256 `7e821a6c90c0884f79f759f5d9be50799ade67bb82515186d922f26032ef43b8`),
  and repo-local Python 3.15.2. Strict `cargo doc --workspace --no-deps --locked`
  passed with the CI `-D missing_docs`, broken-link and invalid-codeblock flags.
  `cargo fmt --all --check`, `git diff --check`, repository policy, polyglot
  documentation inventory and scientific-review freshness checks passed.
  `cargo test --workspace --doc --locked` ran and passed two core doctests;
  the remaining crates reported zero doctests. Both Rust documentation commands
  selected this worktree's Python 3.15.2 interpreter for PyO3.
- The three new focused Rust test targets passed nine tests under
  `cargo llvm-cov --locked -p innovate-core --test native_cli_contract --test
  public_contract_edges --test inference_failure_contracts --lcov`. Their local
  report SHA-256 is `4bbfbc12607953ca684810f86dfe04c353abdfc1af23c59ab0335ad92e343bfc`.
  The retained [coverage run](https://github.com/edithatogo/innovate/actions/runs/43853557986)
  on `ca9739d5` reported 1,653/2,047 changed Rust lines; its LCOV SHA-256 is
  `17ad521baca98c41cebddb06a232e0b32de40e884fce2b72e2f19b1721a89bf4`.
  Mapping documentation-only line shifts and combining the local hits with that
  report gives an estimated 1,901/2,047 (92.87%). This is a local estimate, not
  a full coverage run or a Codecov decision for the eventual PR head.
- `scripts/check_migration.py --release` still exits 1 for incomplete scientific
  migration and acceptance. Release readiness and the dedicated offline-security
  runner remain separate, unwaived gates.
- PR #656 was squash-merged and its branch deleted while this packet was in
  progress. Main commit `57d2916586d107c8affe90355e453abe27f01c26` has the same
  Git tree as `ca9739d5`; the unpushed packet was rebased onto that commit as a
  follow-up to main. Historical PR checks remain historical, and final-head
  hosted documentation and coverage evidence is still required.

## Main browser and CodeQL follow-up — 16 November 2026

- Owner: chat `01a1e3e9-537d-72a0-acc9-56b9ba954680`; worktree
  `/Users/doughnut/.codex/worktrees/innovate-pr656-doc-coverage/innovate`;
  branch `codex/innovate-main-quality-20261116`; base
  `b29a1d27f54fdd08d77881ee999772fceab7c03a`. Initial write scope is this
  evidence entry only. Any source path requires exact diagnostic identification
  and ownership reconciliation before editing. No native authentication or
  tenant object is assigned to this lane.
- PR #659 was closed in favor of draft #671. Its owner retains the overlapping
  browser test files; the successor's hosted documentation workflow passed
  164/164 browser cases on `b2ed945f9ffede3bdd00c29437636c0bf1eba03d`.
  That is evidence for the successor head, not a main-branch pass.
- Main CodeQL reported 129 Rust files extracted without errors and two with
  errors. The exact `b29a1d27` database was downloaded read-only (archive
  SHA-256 `6d36cf420d666e7b6b542d299a1c34feca9549dd8e710c33fdc76447bf3e1fee`)
  and inspected with CodeQL CLI 2.27.4 and the matching `codeql/rust-queries`
  0.1.46 pack. The source-file `ExtractionErrors.ql` query returned zero rows.
  Its summary metric also counts warnings, which are being queried separately.
- Next operation: identify the warnings and their source ownership, then make
  the smallest qualified repair. Scientific release readiness and the dedicated
  offline-security lane remain unchanged and independently unqualified.
- The main database's `ExtractionErrors.ql` returned zero source-file rows.
  Its `ExtractionWarnings.ql` returned two rows: the archived
  `rust_network_petgraph_20261115/evidence/replay_probe.rs` and the active-track
  `scientific_ecosystem_alignment_20261029/evidence/scenario-parity-20261113/
  historical_producer.rs` have no semantic analyzer because neither is in a
  Cargo manifest. The CodeQL summary counts warnings as well as errors. The
  CSV receipts and database archive are in
  `/Volumes/PortableSSD/.innovate-pr668-diagnostics.keHDMl/codeql/receipts/`
  and its parent; the main-warning CSV SHA-256 is
  `ceb7fd6e950c0ee52955924a0bad534350e1783aa39465c509f1184629359462`.
- In an isolated source fixture, a nested manifest eliminated the warning.
  The scenario probe type-checked against historical commit `3681c8a8` with
  its original resolved dependency versions and hashbrown source; its lock
  passed a local RustSec audit. The replay probe type-checked offline against
  baseline `07d6d897` and a separately generated `ff7e077a` candidate overlay
  with the recorded `network.rs` hash and `petgraph` configuration. Its lock
  had no `paste` and passed the local audit. CodeQL 2.27.4 over both temporary
  contexts returned zero extraction-error and zero extraction-warning rows.
  These are isolated-fixture results, not a new main-branch CodeQL run.
- A naive candidate alias to later commit `ead7d5e5` also type-checked but
  changed other core sources and pulled in unmaintained `paste` 1.0.15; it was
  rejected. No analysis manifest, historical source, dependency pin, scanner
  configuration or security exemption has been committed to main by this lane.
  A durable repair must provide the exact historical overlay before GitHub's
  default scan without changing the scientific evidence or undermining the
  concurrent dependency-remediation lane.
- Draft PR #671's current head `070a04424dfb084758d1c0b201bf2980002ea81d`
  passed [hosted documentation qualification](https://github.com/edithatogo/innovate/actions/runs/43869447887),
  including 164/164 browser cases; the PR remains draft and its owner retains
  the affected files. Main's own browser failure is not thereby repaired.
  PR #668's description was updated and read back with the corrected CodeQL
  finding and the successor's exact-head evidence.

### Continuation: scoped browser and historical-scan repair

- The user directed this lane to continue until both issues are resolved on
  main while coordinating with PR #671. The source branch remains
  `codex/innovate-main-quality-20261116` at base `b29a1d27`; the owned
  preflight checkout will temporarily hold an independent historical analysis
  support branch at `ff7e077a`, then return to target preflight use.
- Allowed source scope is the browser qualification helper, its focused tests
  and contracts under `docs/astro-site/{tests/browser,scripts}`, a new
  non-production historical-analysis harness under `tools/codeql-evidence`,
  and this plan. The support branch may add only a standalone candidate-analysis
  manifest against the already recorded historical source. PR #671 retains its
  branch and worktree; any browser backport must preserve its committed source
  identity and be reconciled with its current head.
- Keep active root Cargo manifests, production dependency pins, CodeQL default
  setup, scientific release readiness, coverage thresholds and offline-security
  workflows unchanged. Qualify the historical harness with real source,
  lockfile audits and observed CodeQL diagnostics before proposing integration.
- CodeQL's real-file fixture did not associate manifests under
  `tools/codeql-evidence` with Rust probes outside those manifest directories.
  This lane's allowed scope therefore also includes new standalone
  `Cargo.toml` and `Cargo.lock` files beside the two named evidence probes.
  The existing historical files and their hashes remain unchanged. The replay
  manifest must bind distinct recorded baseline and candidate sources; a pinned
  analysis-only support commit may supply the candidate manifest, without
  changing the root workspace or the scientific/offline gates.
- The scoped browser backport uses the original-response capture helper, its
  focused tests and support fixture exactly from PR #671 head `070a0442`;
  main's archived-version navigation remains present and now waits for each
  navigation's network activity before starting the next. No failed-resource,
  artifact-identity, accessibility, budget or case-count check was removed.
  On Node 26.13.1 and pnpm 12.21.0, 47 JavaScript contract tests passed, the
  site built 202 pages, Astro check reported zero errors or warnings, and
  browser discovery listed 142 cases. Hosted browser execution remains required.
- New standalone manifests and locks beside the two historical probes bind
  the recorded source contexts without modifying the root workspace. The
  replay candidate is pinned to manifest-only support commit
  `b0a60804f6bcd407e2548197301a3976011b08e1`, retained by branch
  `codex/innovate-codeql-historical-candidate`. Both locked Cargo checks passed
  offline; RustSec audited 104 and 71 dependencies without advisories, and
  neither analysis lock contains `paste`. The replay ran 160 full-response
  comparisons and produced SHA-256
  `47e3e6aeec589cc73ba5a17b4518c192a1db328b4e0b57e8e41e8a515ce99c80`,
  exactly matching its historical receipt. This synthetic parity is not
  independent scientific validation.
- A CodeQL 2.27.4 Rust fixture using the published candidate commit and both
  historical manifests returned zero extraction-error and zero extraction-
  warning rows. Full-repository and hosted exact-head results remain required.
  Scientific readiness and the offline-security boundary remain unchanged.
