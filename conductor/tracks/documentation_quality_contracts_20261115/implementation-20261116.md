# Documentation implementation — 16 November 2026

## Authority, source and integration

Dylan authorised implementation of the learning programme, stronger docstrings
and language-native conventions/contracts, obsolete-content removal, and alignment
with the open-science communities identified in GitHub issues. This record
supersedes the planning-only status of the earlier packet and preserves its
unfinished acceptance targets.

Implementation began from main `6c226f92f14f01e771b1490c7fb3a7c77f4bc74a` and was
refreshed onto main `69b674814513e09a683279c679b1584af8bb4214` before final native
qualification. The intervening Kairos dependency update is upstream work; this
increment does not author changes to root Cargo/Python manifests or locks.

The final integration refresh observed main
`18c467e18b71d972060365c68865aa405a69949c` (#660). Its only change from the qualified
native base is `.github/workflows/renovate-config.yml`, updating the Renovate
validator from 44.242.0 to 44.243.0. The final tree carries that upstream change
unchanged with a merge parent. It touches no qualified documentation, native or
notebook inputs, so the recorded main69 execution retains its actual scope.

Code snapshot: `7bd5aa89b0798abaab4132df8096bbc683cbb553`.
Its tree is `c42439d8dae11b1f5dabb5c9fa174ef6a683bf65`; its parents are the qualified
native base `69b6748` and original planning head `76f455fb568639cccb2370720eebcce399b2e158`.
The follow-up commit records the scoped review, source-root correction, foreground
browser harness repair and this reconciliation. Receipts include source-file
hashes where a result covers the working tree rather than that snapshot alone.

The receiving pull request is [#659](https://github.com/edithatogo/innovate/pull/659).
It remains a draft for integration and outstanding qualification. The sole
integrator in [#624](https://github.com/edithatogo/innovate/issues/624) retains
shared CI, root dependencies, registries, merge and publication. No release,
Pages activation or external community submission is performed by this increment.

[PR #656](https://github.com/edithatogo/innovate/pull/656) remained open during
implementation and proposes archiving this track. Integration must retain the
unfinished DOC-01–08 obligations in this active route or a registered successor,
preserve the receipts, and reconcile any directory move before merging #659.
Archival cannot stand in for completion of the curriculum or publication.
The refreshed #656 head `8244d1eb23fec915c0ce93f4da93e3cae44d6ddb` still has the
same documentation archive proposal as `035f2c7`: its registry marks the track
archived/completed while the archived metadata remains `in_progress`. Its added
static adapter work does not resolve this documentation handoff.

## Delivered implementation

### Language conventions and executable reference contracts

- [x] Treat Python docstrings, Rust rustdoc, Julia Markdown documentation and R
  roxygen/Rd as the canonical language sources. Add the current
  [language-conventions guide](../../../docs/astro-site/src/content/docs/maintainers/language-conventions.md).
- [x] Strengthen source/runtime Python parsing to reject duplicate sections and
  named entries. Enforce explicit, type-sensitive convention settings; integers
  cannot silently stand in for Boolean policy values.
- [x] Clarify units, rates, schema 2.0 interchange, result/error envelopes and
  transport in maintained binding documentation and matching R manuals.
- [x] Add Rust FFI safety/ownership guidance and an allocation/free doctest.
  Compiler documentation checks cover all four workspace crates.
- [x] Refresh generated polyglot reference metadata and all 34 API pages.

The public Python inventory contains 55 documented symbols, including 40 public
symbols; installed native documentation covers ten public callables. Julia and R
each expose 24 inventoried public exports. These denominators are different and
are not added together as a single coverage percentage. Executable Python/Julia/R
bodies and Rust FFI behavior are unchanged by these documentation edits.

### Astro discovery and reader routes

- [x] Generate a model atlas and tutorial catalogue from native discovery and the
  existing content/example contract. Typed Astro collections validate the views;
  stale IDs, routes or example bindings fail generation.
- [x] Represent all 16 experimental model keys without inferring combinations from
  independent variant/operation/quantity lists. Family-level rate metadata remains
  visible as declared metadata, not a new scientific support claim.
- [x] Use build-time components, cards, tables, Steps, Tabs and checked source/output
  presentation. Keep catalogues usable as static HTML and retain Starlight search.
- [x] Add focused current lessons for batch execution, forcing and time, network
  scenarios, and Kairos simulation, with actual example source, expected output,
  worked tasks and mechanism-specific limitations.

The current inventory has **14 tutorial pages using 11 distinct ordinary examples**;
all 11 registered examples now have tutorial routes. This is useful teaching
coverage, not complete coverage of all model families or advanced operations.
The model atlas, API references and tutorial catalogue have distinct purposes.

### Public data and optional marimo

- [x] Admit `wdi-aus-internet-20261116`, a frozen WDI/ITU Australian Internet-use
  extract, with original response, query/vintage, CC BY 4.0 attribution, normalized
  CSV, hashes, definitions and limitations.
- [x] Execute a public native Logistic/Gompertz comparison with a chronological
  holdout and independently checked persistence baseline. Preserve the four
  missing years and elapsed calendar time. The baseline's smaller error is a
  deliberate teaching result, not edited away to favour either fitted model.
- [x] Implement `notebooks/public_data_fit.py` as a thin marimo adapter over the
  same ordinary example. A separately locked optional project keeps notebook
  tooling outside default library dependencies.
- [x] Qualify marimo 0.25.2, Python 3.15.2, Polars 2.1.0 and Innovate 0.7.0a1 in the
  actual native environment. Run strict notebook lint, headless defaults, a changed
  cutoff, exact ordinary-example parity and a successful HTML export.
- [x] Add source-only notebook admission checks and negative tests. The execution
  receipt is separate from the source-only build receipt, so a later build cannot
  overwrite execution with an apparently current structural pass.

At cutoff 2015, 23 observations train the curves and nine assess their predictions.
MAE is 15.994 percentage points for Logistic, 12.916 for Gompertz and 8.410 for
persistence. The 2026 data vintage makes this a retrospective holdout, not an
as-of-2015 backtest. Repeated cutoff inspection is exploratory. Recent Internet
use is not cumulative first adoption and does not identify a diffusion mechanism.

The first notebook combines the original N01 compatibility intent with N02-style
public-data teaching. Exactly one notebook is delivered. Headless states at 2015
and 2018 yield nine and six holdout rows respectively; this is not a completed
interactive browser change/reset test. Static export is generated locally but
not published. WASM/Pyodide and a hosted Python service remain unqualified.

### Legacy retirement

- [x] Remove **263 obsolete sources**: 61 historical current-path pages, 80 versioned
  pages, 94 Sphinx sources/templates, 17 legacy notebooks, ten obsolete root guides
  and one legacy generator. The current tutorial index rename is separate from
  that retirement count.
- [x] Retain original Git blob IDs, SHA-256 digests and immutable historical source
  links in the retirement contract, pinned to `6c226f9`. Verify those bytes against
  actual Git history; main's subsequent dependency update did not alter them.
- [x] Render **141 short retirement notices** at former web URLs using Starlight's
  supported custom-page layout. Each gives current guidance and precise historical
  scope, excludes old code, and carries noindex/search-exclusion metadata.
- [x] Remove the obsolete version selector and unused `starlight-versions`
  dependency. Retain current links and immutable source history without adding
  link-validator exclusions or automatic redirects that imply API equivalence.

Historical ADRs, Conductor acceptance receipts and `docs/source/_static` evidence
remain. Retirement removes obsolete active instructions; it does not erase past
decisions or the scientific obligations they recorded.

### Open-science documentation standards

- [x] Add a machine-readable mapping with 12 primary sources and 11 requirements,
  plus a current reader-facing documentation standards page.
- [x] Link actual issue commitments: #398 parent, #399 Software Heritage, #400 RRID
  and #401 JOSS. Delegate registry state to the existing registry-readiness checker.
- [x] Distinguish adopted pyOpenSci/rOpenSci/language guidance from historical
  NumFOCUS and other community aspirations. Ecosystem interoperability issues do
  not establish community admission or membership.
- [x] Reject unsupported statuses, mismatched issue/source mappings, unsafe or
  missing evidence paths and stale page projections. Do not manufacture external
  identifiers, curator decisions, peer review or standards certification.

JOSS research/use and paper readiness, resolving registration identifiers,
maintainer/community commitments and a substantive installed R vignette remain
open. Existing R vignette scaffolding is acknowledged; its presence is not treated
as an executed end-to-end article.

## Verification and observed limits

Full receipts and exact commands are indexed in
[evidence-20261116](evidence-20261116/README.md). Verification uses Python 3.15.2,
Rust/Cargo 1.100.0, Node 26.13.1, pnpm 12.21.0, Astro 7.4.12 and Starlight 0.43.1.
Native checks were rerun after receiving main `69b6748`.

| Check | Observed result | Scope or limitation |
| --- | --- | --- |
| Source contract regressions | 272 tests across 11 suites passed | Includes existing and new tests; no new coverage percentage is inferred |
| Focused native tests | 43 passed | 19 public-data, 16 catalogue and eight convention tests |
| Optional notebook contract tests | 18 passed | Dependency-free source/policy regressions |
| Existing `pnpm test` entry point | 18 Node test cases passed, including five explicit Python suites | Some tests above run again here; counts are not a unique combined total |
| Python ordinary examples | 11/11 exact stdout contracts, 117 executed assertion lines | Installed native Linux environment |
| Public Python doctests | 19 attempted, zero failures | Source/import identity checked |
| Native callable documentation | 10/10 callables match declared source/stubs; zero gaps | Independent native build attestation remains false |
| Rust reference | All four crates pass strict rustdoc compilation | No missing-doc/broken-link/invalid-codeblock warnings accepted |
| Rust doctests | Three passed, zero failed or ignored | Two core, one FFI; tabular has zero, Python cdylib target is unsupported for doctests |
| marimo | Two headless cutoff states, exact parity, strict lint and export passed | Export 54,901 bytes; no interactive browser or WASM qualification |
| Astro build/type check | 211 HTML pages; zero type errors, warnings or hints | 69 current pages, 141 notices and 404 |
| Production artifact | 11,360 local references checked, zero errors | 413 external URLs explicitly not checked by this verifier |
| Retirement | 263 historical identities and all 141 rendered notices checked | Full browser navigation still requires qualification |
| Current content | 69 pages including 34 generated; zero unbound Python blocks or violations | Structural completeness only |
| Scientific review record | 61 declared source inputs pass freshness/consistency | Scoped agent review, not whole-library or human scientific certification |
| Repository policy | Passed, zero findings | Policy inputs and checker unchanged from receiving main |
| Local browser execution | Blocked before any page interaction | Details below; no browser acceptance pass |

Julia and R runtimes were unavailable locally. Their source contracts pass, but
fresh installed execution and the combined-language gate remain open. Historical
hosted Julia/R results retain their original revision and are not reattributed.

### Browser qualification: environment failure retained

The full unfiltered suite discovers 164 desktop/mobile cases. Its first launch
exposed Astro 7's agent-detected preview backgrounding; Playwright's server command
now uses the supported `--ignore-lock` foreground option. The initial attempt
executed zero cases. This is a harness repair, not an application quality result.

The pinned Playwright browser download failed with an invalid ZIP response.
An explicitly recorded system Chromium 156.0.8078.79 fallback was attempted; the
default managed-browser CI path remains unchanged. Chromium then failed before
connecting because `socket()` returned `Operation not permitted` in its process
singleton setup. The repetitive run was stopped after 64 logged setup failures;
zero browser connections/page interactions succeeded and the remaining 100 selected
cases have no reported result. The exact in-flight interrupted count is unknown.
Interrupted execution does not produce a suite pass.

The browser evidence records both attempts and the unavailable final reporter
result honestly. No resource, accessibility, failed-request, identity or threshold
check was disabled, and no restricted socket workaround was attempted. Hosted
browser qualification at the pull-request revision remains required.

## Independent review and resolved findings

Separate agents reviewed source conventions, Astro/contracts, and reader/scientific
interpretation. The scoped review found and resolved three concrete D01 issues:
the loader now rejects an incorrect time unit, the prose explains the retrospective
data vintage, and repeated cutoff exploration is distinguished from untouched
evaluation. Independent native tests show that altering holdout outcomes changes
scores without changing training parameters or held-out predictions.

Worked reader answers cover the new public-data/notebook routes, batch errors,
forcing/time rounding, network adoption versus utilization, and Kairos count/level
conversion. They are agent exercises. Human reader task outcomes, screen-reader
review, exhaustive scientific assessment and external empirical validation were
not performed and remain explicitly unclaimed.

## Remaining work and reconciliation

DOC-01–08 remain in progress. The next bounded work is:

1. Obtain hosted browser/native documentation results on the published PR head;
   repair concrete failures without weakening the gates. Reconcile the #656 active
   track route and shared tech-stack context through the integrator.
2. Finish public build identity and post-publication checks before claiming that
   the live documentation reflects this implementation.
3. Admit D02/D03 for calendar and multiseries workflows; extend diagnostics,
   figures, uncertainty and rolling-origin assessment with appropriate semantics.
4. Develop the remaining competition, complementarity, lock-in, latent, mixture,
   hierarchical, regime and composite lessons against owner-qualified operations.
5. Qualify first-result and substantive tutorial routes in installed R/Julia;
   add the substantive R vignette and notebook interaction/export publication.
6. Run human reader and assistive-technology evaluation, broader scientific review
   and a separate bounded external-reference maintenance check.

The registry hygiene baseline still identifies three unrelated completed tracks
under active paths (ODE, NUTS and CI supply-chain). This increment preserves the
shared registry and records that ownership handoff; it does not archive this
unfinished documentation track to make the hygiene command pass.

Rollback uses ordinary commits for the bounded changes. The retirement manifest
and immutable history retain recoverable sources. A notebook/browser problem does
not require restoring removed numerical APIs or a Python fallback implementation.
