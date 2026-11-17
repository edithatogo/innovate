# Implementation plan

## PR #671 source-review follow-up — 17 November 2026

- Owner: chat `01a1e448-3a28-7653-a681-cd3fdb41f498` in the managed worktree
  `/Users/doughnut/.codex/worktrees/innovate-pr671-ci-repair/innovate`.
  Base: PR #671 head `304789f50541c39f2fd84fed0ca1603d382e568d`.
  Allowed files: `docs/contracts/scientific-pedagogical-review.json`, the
  formatting-only correction in `tests/native/test_documentation_conventions.py`,
  `docs/astro-site/scripts/browser-contract.test.mjs`, the related
  `docs/astro-site/tests/browser/documentation.spec.mjs`, and this plan. Native
  scope: none. Next: verify the reviewed-source follow-up and supply its scoped
  commit after the in-flight owner run ends.
- Compared four changed sources at `f3462528` with their previously reviewed
  bytes and related page claims. The four inputs are unchanged at `304789f5`.
  The review record describes each delta and its source-check limitations;
  historical independent review and qualification receipts remain unchanged.
- A separate fixture candidate at `60986c33756e0a4a5286ffb736f11636dc89f130`
  was superseded by the owner's `304789f5` fixture repair. It is retained as
  local history, not presented as the PR's implementation. Hosted checks for
  the owner's head remain in flight and must not be attributed to this follow-up.
- Focused tests of the received fixture pass locally; Ruff's format check found
  its dictionary comprehension needs formatting. The correction preserves the
  incoming fixture's behavior.
- At CPython 3.15.2, `python -m pytest
  tests/native/test_documentation_conventions.py` passed 14 tests and 15
  subtests. The seven scientific-review contract tests and the 66-input
  freshness check passed; Ruff check, Ruff format check, repository-policy check
  and `git diff --check` passed. These are local source checks, not hosted or
  native execution evidence.
- Continuing beyond the previous freshness failure, `pnpm test` found two
  `ReferenceError` failures: browser-contract tests use the exported
  `retirementEntryRoutes` and `captureInputs` without importing them. Repair the
  test imports and rerun the complete Node suite before qualifying the build.
- Browser case discovery also found the missing `retirementEntryRoutes` import
  in the browser spec. Restore that import before listing the full suite; a case
  listing is not a browser execution or accessibility qualification.
- With Node 26.13.1 and pnpm 12.21.0, the restored imports pass all 56 Node
  tests. The local Astro build produced 211 pages; `pnpm check`, the content
  ratchet and production-artifact verifier passed, with zero artifact errors.
  `pnpm test:browser:list` discovers 164 cases; browser execution remains for
  hosted CI. These local results do not certify scientific or reader acceptance.
- On the separately received head `304789f5`, hosted CI run `44082433990`
  passed the native matrix; Windows reported 443 Python passes, seven skips and
  15 passing subtests. That receipt predates this follow-up and is not an
  exact-head result for it.
- Hosted documentation run `44085118243` at follow-up head `0a5db6d7` passed
  native examples and 162 of 164 browser cases. The two navigation cases failed
  because a merge retained old version-selector interactions immediately after
  asserting that the retired selector is absent. Restore the complete browser
  specification to its previously qualified `070a0442` bytes, retaining the
  current-route, accessibility, retirement, history and search checks. A new
  full hosted browser run is required; the two failed cases are not passes.
- Both the browser spec and its Node contract test now match their `070a0442`
  versions byte for byte. The 56 Node tests, 164-case discovery, artifact
  verification and diff whitespace check pass locally after the restoration.
- PR #671 auto-merge was disabled because live branch protection does not require
  the failing RustSec and release-readiness jobs. The security advisories, 38
  release obligations and queued offline lane remain unresolved.

## Active successor after upstream archival

The merged #656 moved the predecessor to `conductor/archive`. This track keeps
its unfinished DOC-01–08 acceptance active after integration onto main `5821d8a`.
The archived predecessor and all unrelated registry entries remain intact.
New source receives fresh qualification; earlier receipts keep their actual
revisions and limitations. Shared dependency, workflow, merge and publication
ownership remains with the coordinator.

## Published integration follow-up — PR #671

- [x] Publish the implementation as draft PR #671 and close superseded PR #659.
- [x] Integrate upstream `b29a1d` by ordinary merge `b2ed945`, preserving all five
  independent upstream paths and the 16 reviewed implementation Rust blobs.
  Non-documentation source bytes equal upstream in every conflicted file.
- [x] Run the three new upstream core test targets: nine tests pass, with no
  failures, ignored or filtered cases. Retain the exact command and raw log.
- [x] Recheck scientific/notebook source consistency and the production artifact:
  source/artifact fingerprints are unchanged, with 211 pages, 11,361 local links
  and zero errors. Keep the compact comparison separate from full producer JSON.
- [x] Retain the two clean scanner results for ordinary candidate tree `dc4a4ec`;
  do not attribute that scan to the later merge or evidence-only follow-up.
- [x] Record documentation run 43866570433 and native CI 43866570444 as passing
  at `b2ed945`: 164/164 browser cases, both original-response probes, installed
  Python/Julia/R aggregation and all operating-system lanes. The
  [hosted receipts](evidence-20261116/hosted-b2ed945/README.md) preserve actual
  results separately from the unchanged pre-result integration snapshot.
- [ ] Resolve owner-held standalone games, scientific release and offline runner
  holds; retain the confirmed 16-model/22-capability release failure meanwhile.

## Final integrated qualification — base 5821d8a

- [x] Retain fresh receipts for the frozen source at
  `5821d8a38fbc6cbf3822a63474b79c07b18b5bc0`, without reattributing main69 or
  hosted5ee/178 evidence. The [integrated record](evidence-20261116/integrated-5821d8a/integrated-qualification.json)
  binds commands, source/artifact identities and raw-copy provenance.
- [x] Run 286 source tests, 50 focused installed-native tests and 55 Node tests;
  record overlapping invocation counts explicitly.
- [x] Qualify strict rustdoc for the five root-workspace crates and four fresh
  doctests against final frozen source. Retain the 17-file documentation/token
  comparison and the three authorised formatting-comma exception.
- [x] Reinstall the final native package and execute all 11 ordinary examples,
  19 public doctests, native callable documentation and the optional marimo
  notebook's two headless states/parity/export checks.
- [x] Verify 211 HTML pages, 11,361 local links, all retirement notices/history,
  current source contracts and scientific-review record freshness.
- [x] Copy final evidence verbatim and verify original raw hashes before selected
  reversible archival envelopes. Retain notebook HTML as evidence, without site
  publication. Keep earlier receipts historical.
- [!] Qualify standalone games documentation after its upstream dependency
  resolution conflict is resolved by the dependency owner. The strict command
  stopped before rustdoc; doctests were not run and no pins were changed here.
- [x] Obtain current hosted Julia/R receipts and an all-language aggregate pass
  at `b2ed945`; retain the locally incomplete receipt with its original scope.
- [x] Qualify the original-response browser repair and Windows checkout repair
  at `b2ed945`, retaining all 164 browser cases and existing budgets, identities
  and exact frozen-data checks. No documentation/public-data Windows case skips.
- [ ] Complete live revision/publication checks and the remaining human,
  scientific and curriculum acceptance in DOC-01–08.

The earlier read-only main observation `45e33cd` contained only two Renovate
workflow updates after the qualified base. Those receipts remain bound to their
recorded source; the later ordinary merge is qualified separately above.

## Authorised implementation follow-up — 16 November 2026

The maintainer authorised implementation, language-native documentation conventions
and contracts, obsolete-content removal, and open-science issue alignment. The
[implementation record](implementation-20261116.md) records the selected work,
actual verification, partial completion and remaining dependencies. Earlier
planning-only wording below describes the previous increment.

## Learning-programme intake and current status — 16 November 2026

- [x] Reconcile main 06b16671b654b4e51f158bf7d605c4286f156755, open PRs,
  documentation-owner handoff #613 and coordinator issue #624 for this planning
  increment; no open PR was returned at intake.
- [x] Refresh concurrent work before publication: #656 subsequently proposed
  archiving this documentation track. Record the active-successor/directory-move
  reconciliation required before integration; leave its shared registry edits
  with the coordinator. The native/content audit remains pinned to the stated base.
- [x] Audit the maintained tutorial/example inventory and three root legacy
  notebooks; distinguish current examples, public-data gaps and historical APIs.
- [x] Research official marimo/Astro features and publisher dataset terms; record
  compatibility tests, coverage targets, ordered work and ownership boundaries.
- [x] Incorporate the recommendations in the linked programme, curriculum, design
  and dataset records. This checks completion of planning only.
- [~] DOC-01: resolve existing browser delivery failures and publish reproducible
  documentation revision identity through the existing owner/integrator.
- [~] DOC-02: derive the model/operation/tutorial catalogue, reconcile variant
  metadata and classify legacy notebooks before promoting current coverage.
- [~] DOC-03: qualify optional marimo tooling with the exact native environment,
  then execute and export the first notebook through existing contracts.
- [~] DOC-04: admit frozen public datasets and deliver beginner/intermediate
  fitting, model-choice, forecasting and reproducibility lessons.
- [~] DOC-05: use typed Astro collections, Starlight components, scientific figures,
  equations and targeted search to present the shared catalogue and examples.
- [~] DOC-06: complete advanced model/mechanism tutorials against accepted native
  capabilities, including explicit unsupported-case teaching.
- [~] DOC-07: deliver equivalent Python/Julia/R starting routes and practical
  old-to-current migration recipes without claiming unsupported parity.
- [~] DOC-08: complete reader tasks, scoped scientific review, browser/search
  follow-ups and live artifact verification for the expanded learning programme.

Dependencies, ownership boundaries and acceptance are in the
[ordered work packages](learning-plan-20261116.md). The subsequent implementation
delivers bounded parts of every package; none meets its whole-programme acceptance
yet. The current [implementation record](implementation-20261116.md) and
[receipts](evidence-20261116/README.md) bind results to the implemented source.

- [x] Strengthen language documentation conventions and negative-case contracts;
  complete current Python runtime and Rust documentation checks.
- [x] Generate typed Astro catalogues for 16 model keys and 14 tutorial pages,
  covering all 11 registered canonical examples without inferred capability pairs.
- [x] Admit and execute the frozen Australian WDI/ITU Internet-use case, preserving
  calendar gaps, units, provenance and a baseline that outperforms both fitted curves.
- [x] Execute the optional marimo notebook on Python 3.15.2, marimo 0.25.2 and
  Polars 2.1.0; verify two cutoff states, exact ordinary-example parity and HTML export.
- [x] Promote batch, forcing/time, network-engine and Kairos simulation examples
  to focused advanced tutorials with worked tasks and limitations.
- [x] Remove 263 obsolete source files, including 17 legacy notebooks; verify
  immutable history and 141 non-indexed retirement notices at their previous URLs.
- [x] Map issue-linked community documentation requirements to checked evidence,
  retaining unmet registration/submission and substantive vignette obligations.
- [x] Complete scoped independent review and resolve unit, vintage and evaluation
  interpretation findings; retain unperformed human and empirical evaluation.
- [~] Finish browser qualification and hosted revision checks, then hand publication
  and the active-track/archive reconciliation to the sole integrator. Local browser
  launch is blocked by the workspace's socket restriction; no page test passed.

Earlier native execution was refreshed after main advanced to `69b6748`; code
snapshot `7bd5aa8` records that historical implementation. The final integrated
qualification above supersedes its local current-source scope, while preserving
the earlier outcomes and foreground browser-harness correction separately.
Existing scientific, security, release and deployment holds remain.

### Hosted baseline repair — 16 November 2026

- [x] Inspect exact-head run 43853246809 at `5ee03ddc`: Python, Julia and R
  examples pass, but both jobs reject the stale content-baseline policy digest.
- [x] Reproduce the exact ratchet failure locally, review the zero-violation
  inventory and refresh only the baseline digest. Keep accepted debt empty.
- [x] Run the exact source ratchet and 65 isolated content-contract regressions;
  verify unchanged HTML with the updated source fingerprint.
- [x] Qualify the repaired head's installed-language aggregate on run 43855078677;
  all maintained examples pass and the combined gate reports no violations.
- [~] Complete the full browser gate: the same run executes all 164 cases, with
  163 passes and one mobile prefetch-response observation failure. Preserve that
  failure until original-response capture passes at the next head.
- [x] Reproduce the Windows checkout defects; preserve frozen dataset bytes with
  scoped attributes and generate explicit UTF-8 catalogue bytes. The actual Git
  checkout regression and 44 focused cases pass locally.
- [ ] Qualify the response-capture and portability repairs through hosted browser
  and Windows execution without changing thresholds or excluding traffic.

The [hosted follow-up](implementation-20261116.md#hosted-follow-up-and-baseline-binding-repair)
distinguishes actual all-language execution from the failed aggregate policy gate,
the unattempted browser stage and the separate scientific release hold.

### Planning-baseline hosted evidence versus current receipts

At the planning baseline, [documentation run 43804385173](https://github.com/edithatogo/innovate/actions/runs/43804385173)
passed native documentation examples, Astro build/type checks and local artifact
checks. Browser qualification executed 142 cases: 140 passed and two mobile cases
failed (artifact-response mismatch and aborted script request); deployment was
skipped. This supplies current hosted language evidence within that job's scope,
not final browser, public-deployment, scientific or full platform acceptance.
Older receipts below remain attributed to their actual revisions.

The earlier baseline registry-hygiene command exited 1 for three completed tracks
under active paths. Upstream #656 archived them. The integrated successor check
now passes with eight active directories and zero missing, stale, orphaned or
wrongly targeted entries; its receipt is retained separately from the baseline.

## Prior programme history

The [archived predecessor](../../archive/documentation_quality_contracts_20261115/plan.md)
retains the 15 November implementation and verification history. This active
successor carries DOC-01–08 and their unfinished acceptance after upstream #656
archived the earlier directory. Archival is not browser, publication or scientific
completion.
