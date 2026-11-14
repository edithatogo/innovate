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
