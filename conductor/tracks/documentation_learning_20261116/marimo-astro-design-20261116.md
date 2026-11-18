# Marimo and Astro design decisions

This record supports DOC-03 and DOC-05 in the
[learning programme](learning-plan-20261116.md). Research is current to
16 November 2026. The original recommendations were researched against
06b16671b654b4e51f158bf7d605c4286f156755; the implemented increment was refreshed
onto main 69b674814513e09a683279c679b1584af8bb4214. The status below supersedes
the original provisional adoption decision while retaining unfinished acceptance.

## Implemented decision and qualified boundary

The optional native notebook is now implemented in `notebooks/public_data_fit.py`.
Its separate `pyproject.toml` and `uv.lock` pin marimo 0.25.2 and install this
checkout's Innovate package. Python 3.15.2, Polars 2.1.0 and Innovate 0.7.0a1 were
actually installed and executed. The notebook calls the canonical public-data
workflow; default cutoff 2015 and changed cutoff 2018 both reproduce the ordinary
example exactly and yield different held-out predictions. Strict marimo lint and
HTML export pass; the recorded export is 54,901 bytes. Native binary, fixture,
source and environment identities are in the
[notebook receipt](evidence-20261116/notebook-qualification.json).

The first notebook combines the original N01 compatibility intent with the
N02-style observed-data comparison. It is not two completed notebooks. The
synthetic first-result script remains available. Interactive browser change/reset,
the full original N01 unit/rejection controls and static-preview publication are
still pending. No WASM, Pyodide or hosted Python service is qualified.

Astro now uses typed learning collections and build-time model/tutorial components
derived from native discovery and the content contract. Starlight components
present the learning routes; its native search component supplies the search UI.
The obsolete version selector and `starlight-versions` dependency are removed.
Supported custom pages render 141 short migration notices without old code or
search indexing. New client-side frameworks or notebook runtimes are not loaded
on ordinary documentation pages. Scientific figures/math rendering and richer
facets remain later work, subject to the existing browser budgets.

[ADR 0008](../../../docs/adr/0008-current-documentation-and-learning-resources.md)
records the scoped decision. Shared tech-stack/root dependency reconciliation
remains with the integrator. The complete current acceptance record is in
[implementation and verification](implementation-20261116.md).

## 1. Native Python learning: rationale and remaining acceptance

Marimo is a good candidate for parameter exploration, fit diagnostics, scenario
comparison and reproducible data preparation. Its Python-file format is suitable
for version review, and its reactive execution can connect a control to the
native request and its displayed result. Retain readable Astro lessons and ordinary
examples so the notebook interface is optional.

The official [0.25.2 release](https://github.com/marimo-team/marimo/releases/tag/0.25.2)
adds Python 3.15 support. The completed compatibility run qualifies the recorded
Linux environment only; the release announcement itself is not that evidence.
Other platforms require their own dependency/import/native execution results.

### Execution modes

| Mode | Recommendation | What it proves and requires |
| --- | --- | --- |
| Local native marimo editor/app | First interactive target | Runs the installed Rust-backed Python API; requires qualified project environment and exact source |
| Executed static HTML | Initial GitHub Pages publication target | Shows recorded native results; Python controls do not recompute. Export and browser rendering must both pass |
| Native server-hosted app | Possible later increment | Can execute native code remotely; requires a hosting/runtime owner, operating limits and a separate deployment decision |
| WASM/Pyodide export | Deferred compatibility investigation | Needs a working Innovate native extension for that browser ABI, its dependency chain and qualified results |
| `mdx-marimo` in Astro | Deferred for live native computation | Official integration exists, but browser hydration uses Pyodide and inherits the same native-compatibility requirement |

Official references: [static export](https://docs.marimo.io/guides/exporting/static_html/),
[WASM export](https://docs.marimo.io/guides/exporting/webassembly_html/) and
[mdx-marimo, including its Astro example](https://github.com/marimo-team/mdx-marimo).
Availability of Polars in a browser does not establish availability of Innovate's
extension. A precomputed scenario selector can be useful, but must say that it
selects recorded native outputs rather than fitting or simulating live.

### Authoring and scientific boundaries

- Keep notebook source as reviewable `.py` files. Factor shared scientific calls
  into the canonical checked example or helper and keep notebook cells as thin
  presentation adapters. Avoid separate copies of the same workflow in MDX,
  notebook and script. Register the helper/source dependency hashes as well.
- Use public native operations and supported Polars/Arrow inputs. Do not restore
  pandas-era tutorials, implement a Python fallback solver, or pull Voiage's
  economic work into Innovate. Independent analytic expected values may be used
  as explicitly labelled tests, not as the application's numerical engine.
- Create fresh request objects and new result variables for reactive changes.
  Do not mutate `FitResult.envelope` or a shared dictionary in another cell:
  [marimo does not track arbitrary mutations](https://docs.marimo.io/guides/reactivity/).
- Provide deterministic defaults, seeds where applicable, units, admissible
  control ranges and informative native rejection examples. Repeat a control
  change/reset to demonstrate that the displayed result is current.
- Trigger expensive fits, MCMC and simulation batches with an explicit form/run
  control; show the inputs used by the displayed result. Any cache key must include
  relevant inputs, code, native artifact and dataset identity.
- Keep novice lessons short and purposeful. Provide advanced diagnostics in a
  separate route or disclosure rather than making every notebook comprehensive.

### Environment and execution contract

The implemented choice is a separately locked optional `notebooks/` project,
using the same checkout and native package. This replaces the proposed root
tooling-group edit and keeps marimo outside default library dependencies. The
integrator retains shared tech-stack reconciliation. The official
[package-management guide](https://docs.marimo.io/guides/package_management/)
distinguishes projects and isolated sandboxes; a standalone sandbox must explicitly
install the intended pinned native artifact instead of accidentally using another
wheel or missing the local extension.

Qualify the following once for N01, then apply the contract to each notebook:

1. Clean environment installation and native import identity.
2. Notebook lint, explicit collected tests and script execution with default inputs.
3. A known-error control that fails the runner/export, and a relevant invalid-input
   case whose expected native error is handled in the lesson.
4. Interactive control change and reset, checking outputs and stale-state behaviour.
5. Executed export with source, environment, native artifact, fixture and output
   hashes, plus timeout and memory/size bounds.
6. Rendered output checks for expected numbers/figures, missing assets, keyboard
   use, mobile layout and reproducibility instructions.

Define deterministic headless inputs for scripts and static export through the
same request-building path as the interactive notebook. Required computations
must run without waiting for a form/button submission. Assert that the expected
native results were produced and match the interactive default configuration;
a successful exit or rendered notebook shell alone does not qualify execution.

The repository's ordinary pytest selection does not automatically prove that
new notebook tests are collected. Use an explicit collection command and record
the count; follow marimo's
[testing guidance](https://docs.marimo.io/guides/testing/). Match numerical checks
to scientific tolerance or deterministic invariants; do not require every HTML
byte to be numerically reproducible when metadata legitimately varies.

The documented export command is:

```sh
marimo export html notebook.py -o notebook.html
```

Run it inside the qualified environment using the exact accepted source. The
official static-export guide warns that cell failures can still produce HTML
with a nonzero exit status. Publication must fail on that status; file existence
is insufficient. No error-swallowing wrapper is permitted.

### Astro delivery of notebooks

Start with a standalone static preview linked from the lesson, a source download,
environment instructions and an equivalent Astro explanation/table/figure.
Static marimo exports can use JavaScript for rendering, so they are not a
no-JavaScript substitute for the main lesson. A later iframe must have a useful
title, keyboard/focus behaviour, responsive height, lazy loading and a direct
fallback link, and must pass the normal resource/base-path and size checks.

Inspect everything included in the export, including code, outputs, input records
and any embedded notes. Preserve dataset attribution and avoid publishing material
that is not admitted for the intended distribution. This is especially relevant
to the restricted optional D05 case.

The original N01 proposal adapted the existing Logistic result for control/units
exploration, followed by N02's observed-data fit. The implemented first notebook
uses the admitted D01 comparison as described above. Later notebooks can present generations,
uncertainty, forcing/network scenarios, provenance and batch diagnostics once the
corresponding scientific lessons and native operations are qualified.

## 2. Make fuller use of the existing Astro/Starlight site

The [Astro configuration](../../../docs/astro-site/astro.config.mjs) provides
Starlight, MDX/source examples, Pagefind with native Starlight search and the
existing accessible-label repair, optional Algolia, link validation and
`/innovate/` static base-path handling. The
[content schema](../../../docs/astro-site/src/content.config.ts) retains
`docsLoader`/`docsSchema` and now adds typed model/tutorial learning collections.
Expressive Code, build checks and Playwright/axe are existing assets. The obsolete
version archive and selector have been retired under the checked manifest.

### Feature decisions

| Capability | Concrete use in Innovate | Dependency or acceptance condition |
| --- | --- | --- |
| Typed content collections | Validate tutorial level, tasks, prerequisites, example/dataset IDs and model/operation relationships | Derive capability and qualification fields from existing contracts; one status authority |
| Generated static routes/tables | Model atlas, tutorial catalogue, dataset catalogue and related-content links | Every discovered ID has a route/disposition; unknown or stale IDs fail validation |
| CardGrid/LinkCard and Steps | Task-led entry points and ordered install/fit/reproduce lessons | Preserve readable headings and useful links without client JavaScript |
| Tabs with synchronized language selection | Equivalent Python/Julia/R examples | Each language example executed; distinct limits explained rather than hidden by a tab |
| Asides, badges and purposeful FileTree | Assumptions, native errors, version/support state and bundle contents | Derive support labels; avoid decorative status badges or warnings without a teaching purpose |
| Existing Expressive Code | File names, focused line markers, migration diffs and reliable copy actions | Keep complete runnable source accessible and exact-source validation intact |
| Shared MDX components | Checked source, expected result, scientific figure and reproducibility panel | Extend current source-binding checks and negative cases for every new component shape |
| Build-time equations | Consistent model cards and parameter definitions | Review and lock `remark-math`/`rehype-katex` increment; local assets, overflow and accessible output checks |
| Scientific SVG and appropriate image assets | Observed/fitted/forecast curves, residuals, intervals and model comparisons | Generate from checked outputs; captions, units, labels, provenance and accessible alternative data |
| Build-time Mermaid where useful | Branching model choice or provenance relationships | Qualified build dependency; use static output rather than a global diagram runtime |
| Pagefind metadata and filters | Find lessons by reader task, model, language and level | Preserve retirement exclusion and verify actual result relevance; richer facets remain pending |
| Small progressive browser interactions | Catalogue filters or selection among precomputed native scenarios | Clear execution mode, static fallback, keyboard/zoom checks and resource budgets |
| Edit/source links and migration routes | Exact documented package/source/build identity and current migration navigation | Legacy version selector removed; full public build identity remains pending |

Use the existing native residual-diagnostic SVG facility where appropriate and
generate other figures from native outputs. Select a plotting dependency only
when the existing renderers cannot meet a concrete need, through the normal
optional-tooling decision. Scientific outputs remain numerical artifacts with
reviewable data, rather than manually drawn illustrations presented as evidence.

Search journeys should include reader language such as “fit my data”, “which
model”, “uncertainty”, “new generations”, “missing values”, “units”, “R example”
and “network simulation”. Check that historical pages cannot masquerade as the
current answer. Search reachability alone does not establish result usefulness.

### Features to defer or use only after a demonstrated need

Static GitHub Pages has no application server for Astro actions, server islands,
sessions or on-demand endpoints. These require an appropriate adapter/runtime;
they are not prerequisites for richer scientific documentation. Revisit them
only for an owned server-hosted notebook service or another concrete use case.

Do not introduce a UI framework merely to display static cards/tables or add
global view transitions/aggressive prefetch while existing resource-lifecycle
failures remain unresolved. Keep Pagefind as the default; a hosted search service
is optional. Internationalization needs maintained translations and tested
scientific terminology before enabling a language switcher.

### Performance and accessibility are acceptance criteria

The current [browser contract](../../../docs/astro-site/scripts/browser-contract.mjs)
sets decimal-byte budgets of 25,000,000 for the artifact, 512,000 per non-HTML asset,
1,500,000/40 requests for initial navigation and 2,000,000/65 requests for the
search journey. HTML has its separate configured limit. These are decoded-body
limits, not compression or speed claims.

Measure notebook exports, math fonts and generated figures against those actual
budgets. If a justified change is needed, propose the measured exception to the
owner; do not quietly exclude the new assets. Retain the normal Chromium gate and
add a bounded representative cross-engine/assistive-technology review. Theme, zoom,
tables, math overflow and code-copy behaviour matter for scientific readers.

## Primary feature references

These sources describe features; they do not certify compatibility with Innovate's
locked versions. Recheck the installed version before implementation.

- [Astro content collections](https://docs.astro.build/en/guides/content-collections/).
- [Astro islands](https://docs.astro.build/en/concepts/islands/) and
  [server islands](https://docs.astro.build/en/guides/server-islands/).
- [Astro GitHub Pages deployment](https://docs.astro.build/en/guides/deploy/github/).
- [Starlight authoring](https://starlight.astro.build/guides/authoring-content/),
  [code](https://starlight.astro.build/guides/authoring-content/#code-blocks),
  [Steps](https://starlight.astro.build/components/steps/) and
  [Tabs](https://starlight.astro.build/components/tabs/).
- [Starlight search](https://starlight.astro.build/guides/site-search/) and
  [Pagefind filtering](https://pagefind.app/docs/filtering/).
- [Math rendering plugins](https://github.com/remarkjs/remark-math) and
  [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli).
