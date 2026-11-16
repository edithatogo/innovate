# Documentation implementation — 16 November 2026

## Authority, source and integration

Dylan authorised implementation of the learning programme, stronger docstrings
and language-native conventions/contracts, obsolete-content removal, and alignment
with the open-science communities identified in GitHub issues. This record
supersedes the planning-only status of the earlier packet and preserves its
unfinished acceptance targets.

Implementation began from main `6c226f92f14f01e771b1490c7fb3a7c77f4bc74a` and was
refreshed onto main `69b674814513e09a683279c679b1584af8bb4214` before initial native
qualification. The intervening Kairos dependency update is upstream work; this
increment does not author changes to root Cargo/Python manifests or locks.

An earlier integration refresh observed main
`18c467e18b71d972060365c68865aa405a69949c` (#660). Its only change from the qualified
native base is `.github/workflows/renovate-config.yml`, updating the Renovate
validator from 44.242.0 to 44.243.0. The preserved #659 history carried that change
unchanged with a merge parent. That earlier refresh touched no qualified
documentation, native or notebook inputs, so its main69 receipt keeps its scope.

Code snapshot: `7bd5aa89b0798abaab4132df8096bbc683cbb553`.
Its tree is `c42439d8dae11b1f5dabb5c9fa174ef6a683bf65`; its parents are the qualified
native base `69b6748` and original planning head `76f455fb568639cccb2370720eebcce399b2e158`.
The follow-up commit records the scoped review, source-root correction, foreground
browser harness repair and this reconciliation. Receipts include source-file
hashes where a result covers the working tree rather than that snapshot alone.

The initial draft [#659](https://github.com/edithatogo/innovate/pull/659) preserves
its published implementation snapshots and hosted failure/repair history. During
qualification, upstream #656 was merged at `57d2916`, followed by #667 at
`5821d8a38fbc6cbf3822a63474b79c07b18b5bc0`. The successor implementation is prepared
on the isolated `docs/documentation-learning-20261116` branch from that main.
The earlier draft's history is preserved; a new draft receives the complete
integrated source as an ordinary commit so security scanners inspect the changes.

The upstream merge archived `documentation_quality_contracts_20261115` despite
its unfinished browser, curriculum and publication acceptance. This active
`documentation_learning_20261116` successor preserves DOC-01–08, the current
implementation and all original receipts. Its one proposed registry entry leaves
the upstream archive and every unrelated track intact. The sole integration
coordinator in [#624](https://github.com/edithatogo/innovate/issues/624) retains
shared dependencies, workflows, registry integration, merge and publication.
No release, Pages activation or external submission is performed here.

The received native changes require fresh execution rather than reusing the
main69 or head178 results. The updated Rust documentation contract distinguishes
five workspace crates from the excluded standalone `innovate-games` crate.
Documentation-only completion describes the new typed Rust API without promoting
it into the unchanged 16-key dispatcher catalogue. The regime guide is aligned
with the upstream PELT implementation and its 100,000-observation bound, while
retaining the separate 1–2000 jump limit and scientific qualification limits.

Seven source-file checksums in a newly retained artifact receipt triggered the
security scanners' generic API-key heuristic. Each matched value was independently
verified against its original source. Retained evidence uses an explicit,
lossless SHA-256 representation where needed, preserving paths and original
record identity. Scanner rules, allowlists and thresholds are unchanged. The
new ordinary integration commit will receive a complete fresh scan.

## Final integrated qualification

Fresh source and native qualification covers the frozen integrated working tree
on **`5821d8a38fbc6cbf3822a63474b79c07b18b5bc0`**. The
[integrated record](evidence-20261116/integrated-5821d8a/integrated-qualification.json)
indexes the exact commands, input hashes, original receipts and separate limits.
The working-tree source maps identify tested bytes before publication; the base
commit alone does not identify the documentation changes.

| Check | Final observed result | Scope or limitation |
| --- | --- | --- |
| Source regressions | 286 passed: 272 original cases plus 14 convention/workspace-scope cases | Dependency-free source qualification |
| Focused installed-native selection | 50 passed, zero failures or skips | 20 public-data, 16 catalogue and 14 convention/workspace-scope cases |
| Existing `pnpm test` | 55 Node tests passed | Includes explicit Python wrappers; counts overlap other invocations |
| Python examples and public doctests | 11 examples with exact stdout and 117 assertion lines; 19 doctests with zero failures | Freshly reinstalled final native Linux environment |
| Native callable documentation | Ten callables, matching installed source, zero gaps | Independent native-binary build attestation remains false |
| Root-workspace rustdoc | Strict missing-doc, broken-link and code-block checks passed for all five crates | Includes `innovate-kairos`; excludes standalone games |
| Fresh final Rust doctests | Four passed, zero failed, ignored or filtered | Two core, one FFI and one Kairos; tabular has zero and Cargo does not support the Python cdylib target |
| Optional marimo workflow | Strict lint, two cutoff states, exact ordinary-example parity and HTML export passed | Native Python execution only; export is retained evidence, not site publication |
| Astro build and check | 211 HTML pages; zero errors, warnings or hints | Browser rendering and interaction remain separate |
| Production artifact | 11,361 local links checked, zero errors | 413 external URLs were not checked |
| Content and retirement | 69 current pages, 34 generated API pages, 11 examples and zero structural violations; all 141 notices and 263 removed-source identities verified | Source and static-output checks |
| Scientific review record | 66 declared source hashes matched | Record consistency, not scientific or pedagogical certification |

These counts describe separate invocations and must not be added as unique test
coverage. The 14 convention cases, for example, also appear in the focused
selection and a Node wrapper. The source/artifact fingerprints are respectively
`cf3ea0712e66f7512c76f248d1eddfab75edf1bb14c88e3466af5a742b9b21ca` and
`adfc2087fa55fa9e3605da5db9604650d8a2531865c1e37a4c11ed2365a4169b`.
Contemporaneous hashes do not independently establish build provenance.

The strict source-only content command exits 1 with zero violations because it
does not execute examples. The fresh local execution aggregate also exits 1:
Python examples and doctests pass, but no current Julia/R runtime receipts were
supplied. It records `current_python_content_complete: true`,
`current_content_complete: false` and `gate_passed: false`. The historical hosted
178 all-language pass remains valid for that earlier source only.

Documentation completion covers 17 Rust source files. Sixteen preserve exact
non-documentation bytes; `likelihood.rs` additionally contains three explicitly
authorised rustfmt trailing commas, with equal rustfmt-normalised stripped source.
The token evidence includes negative controls. The earlier Rust completion record
retains its actual doctest timing; the subsequent
[final-runtime doctest log](evidence-20261116/integrated-5821d8a/final-runtime/rust-doctests.log)
records the fresh four-test run against the frozen source.

The separate games documentation command exits 101 before rustdoc: the received
Kairos dependency requires `thiserror ^2.0.21`, while Paignia `5103675` pins
`thiserror =2.0.18`. Its doctests were not run. The
[games record](evidence-20261116/integrated-5821d8a/games/games-documentation-20261116.json)
preserves unchanged source, manifest and lock identities. No dependency repair
or games qualification is inferred from the successful root workspace gate.

The original-response capture repair and its mandatory homepage prefetch probe
are implemented and regression-tested. Discovery still selects all 164 browser
cases. Actual managed-browser and Windows execution at the integrated head remain
hosted gates; the local checkout regression does not establish Windows runtime
acceptance. Human reader evaluation, broader curriculum/scientific acceptance,
external submissions and live publication remain unfinished.

All 60 retained raw receipt files were byte-compared with their originals before
selected reversible archival envelopes were applied. The copy manifest records
original raw hashes; any stored archive has its own byte identity, indexed
separately. See the [archive format](evidence-20261116/archive-format/README.md).
The retained 54,901-byte notebook HTML is a local execution artifact and is not
included in the published site.

A subsequent coordinator read-only refresh observed main
`45e33cd6344e49351ce116c816b4d0a7edeebbc7`. Its two Renovate commits after `5821d8a`
change only `.github/workflows/renovate-config.yml`, from validator 44.243.0 to
44.243.2. The qualified checkout remains `5821d8a`; no rebase, rebuild or
reattribution to the later main was performed.

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
  The initial compiler receipt covers the four crates present at main69. The
  integrated successor expands the declared workspace to five crates, with fresh
  documentation completion and qualification recorded separately below.
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

## Initial verification and observed limits — main69 source

Full receipts and exact commands are indexed in
[evidence-20261116](evidence-20261116/README.md). Verification uses Python 3.15.2,
Rust/Cargo 1.100.0, Node 26.13.1, pnpm 12.21.0, Astro 7.4.12 and Starlight 0.43.1.
Native checks in this historical table were rerun after receiving main `69b6748`.
They are not reattributed to the subsequently received main582 native source.

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

Julia and R runtimes were unavailable locally. Their local source checks are
separate from the later hosted results below: run 43855078677 qualifies the
installed examples and combined-language gate at `178779bd`. It does not complete
the wider R/Julia tutorial programme or reattribute historical receipts.

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

### Hosted follow-up and baseline binding repair

After publication of `5ee03ddc`,
[documentation run 43853246809](https://github.com/edithatogo/innovate/actions/runs/43853246809)
executed the maintained language examples successfully: all 11 Python examples
and 19 API doctests, seven Julia examples with ten assertions on Julia 1.13.4,
and six R examples with 11 assertions on R 4.6.2. Rust documentation/doctests,
the Astro build/type check and source-contract tests also passed. These hosted
results close the missing installed-language **example execution** evidence at
that source; they do not deliver the wider R/Julia curriculum.

Both aggregate jobs nevertheless failed the policy ratchet because
`content-baseline.json` still named the old content-policy digest. The new
69-page inventory had zero violations, and native aggregation reported no
execution failure, but its final `gate_passed` remained false. Browser discovery,
installation and tests were skipped; the subsequent missing browser-upload
failure was secondary. Deployment was skipped. See the
[exact-head hosted record](evidence-20261116/hosted-5ee03ddc.json).

The repair updates only the reviewed baseline's policy digest to
`354ee38da7313908287f953031d8b4dc4c367d83193b323f5de2cdf23ebf9e61`, retaining
an empty accepted-violations list. The exact source ratchet reproduced failure
before the change and passes afterwards; all 65 content-contract regressions pass
with isolated, dependency-free discovery. No rule, threshold or runtime
requirement changes. The [repair record](evidence-20261116/ratchet-repair.json)
preserves both results. The artifact verifier was refreshed without rebuilding
unchanged HTML; its artifact hash is identical and only the baseline source hash
changes. Hosted qualification was repeated at `178779bd`, as recorded below.

Workflow/policy lint and secrets checks passed on `5ee03ddc`. Release Readiness
correctly remains failed for the unchanged scientific migration ledger: 16 model
and 22 feature-area obligations still block release. That hold is unrelated to
the documentation repair and is preserved.

### Hosted aggregate pass and observed prefetch failure

[Run 43855078677](https://github.com/edithatogo/innovate/actions/runs/43855078677)
checks head `178779bd` at GitHub merge checkout
`5f0cc853b48a7c14479f71ccf5183a6d1265e582`. The native documentation job passes,
including all 11 Python examples and 19 doctests, seven Julia examples with ten
assertions, and six R examples with 11 assertions. The combined content gate is
now `gate_passed: true` with zero violations and no stale policy binding. Full
source, build, type, artifact and Rust documentation checks also pass.

Managed Chromium 159.0.8154.6 executes the complete 164-case suite: **163 pass and
one fails**, with no skips or retries. The failure is the mobile dark-entry case
for the native migration page. A successful homepage prefetch transfers 8,051
encoded body bytes, but Playwright exposes zero decoded bytes to the artifact
comparator. The strict comparison correctly remains failed. The case's two axe
scans report no violations or incomplete rules, and its recorded transfer budget
is within the existing limit; neither fact establishes complete browser acceptance.
Deployment is skipped.

The [exact-head hosted record](evidence-20261116/hosted-178779bd.json) retains the
full case-outcome inventory, report and trace hashes, artifact IDs and the failure
details. Separate unchanged native execution receipts preserve the successful
installed-language results. The browser repair must capture the original
speculative response, require unambiguous request identity and completed delivery,
and retain its decoded bytes and request in the existing budgets. A second fetch,
empty-body exemption, skipped prefetch, or disabled site feature cannot qualify it.

### Windows checkout portability

The Windows native matrix at `178779bd` repeats five concrete failures: generated
catalogue bytes change under platform newline translation, and four public-data
tests reject the checkout-translated CSV against its frozen checksum. Linux and
macOS native jobs, and all three Julia/R platform jobs, pass. The failing lint and
unit-test alias jobs inherit the native matrix result rather than identify a
separate code failure.

The repair writes catalogue output as explicit UTF-8 bytes and pins only the three
frozen snapshot files to Git's `-text` attribute. Dataset bytes, recorded hashes,
loader verification, ordinary example and scientific interpretation remain
unchanged. An actual temporary Git checkout with `core.autocrlf=true` converts an
ordinary control file to CRLF while preserving the frozen files byte for byte.
That regression fails without the attributes and passes with them. All **44
focused top-level cases** pass: 20 public-data, 16 catalogue and eight convention
cases, with zero failures or skips. Scoped Ruff checks and source freshness also
pass. See [the portability repair](evidence-20261116/portability-repair.json).
Actual Windows qualification of the repair still requires the next hosted head.

### Outstanding programme acceptance

DOC-01–08 remain in progress. The next bounded work is:

1. Obtain hosted browser/native documentation results on the published PR head;
   repair concrete failures without weakening the gates. Integrate this active
   successor and shared tech-stack context through the coordinator.
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

The earlier registry hygiene baseline identified three completed tracks under
active paths. Upstream #656 resolved those directory moves. The integrated
successor check passes with eight registered active directories and zero findings.
The retained baseline keeps its original result; current completion is not
inferred from an archive label. This successor remains `in_progress`.

Rollback uses ordinary commits for the bounded changes. The retirement manifest
and immutable history retain recoverable sources. A notebook/browser problem does
not require restoring removed numerical APIs or a Python fallback implementation.
