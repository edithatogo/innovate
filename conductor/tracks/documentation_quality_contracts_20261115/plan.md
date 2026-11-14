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
- [~] Open a focused PR and verify exact-head hosted checks; preserve coordinator merge ownership.

## Full-quality acceptance remains open

- [x] Reach zero structural debt for current pages and execute all ten current Python examples without accepted exceptions.
- [x] Complete compiler-verified Rust public documentation and executable rustdoc examples.
- [ ] Qualify every claimed supported language tutorial against its installed artifact.
- [ ] Verify browser accessibility, keyboard/search/mobile behavior and performance budgets.
- [ ] Review scientific claims and tutorial teaching quality against primary references and reader tasks.
- [ ] Verify the qualified Astro artifact at the public URL after authorized deployment activation and integration.

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
