# Documentation learning programme — 16 November 2026

## Authority, scope and status

Dylan requested that recommendations for marimo notebooks, fuller Astro use and
comprehensive current tutorials with public datasets be incorporated into the
tracks, then authorised implementation of those recommendations, language-native
documentation contracts, obsolete-content removal and open-science alignment.
This extends documentation_quality_contracts_20261115.
The [specification](spec.md) and [plan](plan.md) remain authoritative entry points.

Planning source: **06b16671b654b4e51f158bf7d605c4286f156755**. The implementation
was refreshed onto main **69b674814513e09a683279c679b1584af8bb4214** and captured
in code snapshot **7bd5aa89b0798abaab4132df8096bbc683cbb553**. Dates use
Australia/Brisbane unless a receipt explicitly records UTC. DOC-01 through DOC-08
are now **partly implemented**; none has full acceptance. See the
[implementation record](implementation-20261116.md) and
[qualification receipts](evidence-20261116/README.md) for exact execution scope.

The existing documentation owner retains the #613 content/contracts/browser scope.
The sole integration coordinator in [issue #624](https://github.com/edithatogo/innovate/issues/624)
retains root manifests/locks, shared CI, common registries and integration. The
[documentation handoff](https://github.com/edithatogo/innovate/pull/613#issuecomment-6890734373)
remains relevant. New work is selected in bounded packets after fresh ownership
and head checks; the absence of an open PR does not release another owner's scope.

This increment stays inside the existing track so the common registry already
reaches it through the track index. Shared root manifests/locks, workflows,
registries and publication remain integrator-owned. The optional notebook uses a
separate locked `notebooks/` project under
[ADR 0008](../../../docs/adr/0008-current-documentation-and-learning-resources.md);
it does not change default library dependencies or shared CI.

### Delivered increment and next acceptance

| Package | Implemented and checked | Still required for the package |
| --- | --- | --- |
| DOC-01 | Production artifact/source identities; 211 built pages and checked links; foreground preview repair | Browser launch in a permitted environment, hosted revision result, public build identity and post-publication check |
| DOC-02 | Typed discovery for 16 model keys, 14 tutorials and 11 canonical examples; drift checks; complete legacy inventory | Full variant/operation/quantity reconciliation, explanatory model chapters and long-guide restructuring |
| DOC-03 | One locked native marimo notebook; real execution at two cutoffs; ordinary-example parity; strict lint and successful export | Interactive browser change/reset, published static preview, further notebooks; WASM remains a separate investigation |
| DOC-04 | D01 frozen WDI/ITU case; chronological fitting/comparison, independent persistence baseline, offline provenance checks | D02/D03 admission, calendar/multiseries courses, broader diagnostics and rolling-origin assessment |
| DOC-05 | Typed Astro collections, generated cards/tables, Starlight components/search and migration routes | Qualified browser behaviour, scientific figures/math, useful search facets, public artifact verification |
| DOC-06 | Four focused lessons: batch, forcing/time, network-engine and Kairos simulation | Remaining coupled/wrapper/inference chapters and accepted operation coverage |
| DOC-07 | Stronger Python/Rust/Julia/R conventions and contracts; Python execution and Rust compilation; checked migration notices | Fresh installed R/Julia course execution and equivalent first-result pathways; substantive R vignette |
| DOC-08 | Scoped independent source/claim review, worked answers, resolved unit/vintage/evaluation findings and separate evidence scopes | Human reader/assistive-technology evaluation, broader scientific assessment, external-link maintenance and publication |

The marimo sequence was deliberately revised: the first notebook combines the
N01 interface spike with the new N02-style public-data comparison. It calls the
same `native_public_data.py` workflow as the Astro lesson. This is **one notebook**;
the original synthetic Logistic notebook, full N01 control/reset acceptance and
the later notebook curriculum have not all been delivered.

### Concurrent archive proposal

The planning refresh found [PR #656](https://github.com/edithatogo/innovate/pull/656),
opened after intake, proposing to move this track to the archive. It remained
open during implementation. This increment does not edit that PR or the shared
registry. Before integration, the coordinator must reconcile the archive move:
retain an active documentation track for DOC-01–08 or transfer this new plan and
its index links to an active registered successor. Preserve the earlier receipts
and the new unfinished obligations; archival is not implementation acceptance.
If #656 lands first, rebase and resolve its directory move before integrating
this packet, then refresh any changed native capability denominators.

## Assessment and decisions

1. The planning baseline contained eight tutorial pages, six distinct example
   files and ten registered Python examples. The implementation now has fourteen
   tutorials, eleven distinct examples and an observed-data fitting route. The
   [coverage audit](tutorial-coverage-20261116.md) separates this improvement from
   the remaining beginner-to-advanced curriculum.
2. Adopt marimo as an **optional native Python learning interface**, with the first
   exact environment executed successfully. Local interactive instructions are
   present; publishing the executed static preview remains pending. Browser WASM and hosted
   notebook services require separate demonstrated compatibility/ownership.
3. Use Astro's existing content collections and Starlight components more fully.
   Generate navigation and capabilities from shared identifiers and contracts.
   Retain static GitHub Pages delivery, Pagefind and the existing quality stack.
4. Provide beginner, intermediate and advanced routes covering all declared model
   families and meaningful supported operations. Every exposed capability gets a
   current example/reference route or an explicit unresolved/unsupported record.
   A giant Cartesian product of repetitive notebooks is not the coverage target.
5. Add public data for empirical analysis, with retained synthetic oracles for
   correctness and difficult cases. Dataset admission is per artifact; public
   availability alone is not redistribution or scientific-validation evidence.
6. Preserve Rust scientific ownership, Polars tables, the maintained three language
   bindings, and Kairos/Voiage boundaries. Notebooks call the public native API;
   they do not implement missing numerical, simulation or economic functions.

See [design decisions](marimo-astro-design-20261116.md) and the
[dataset shortlist](public-datasets-20261116.md) for sources and acceptance detail.

## Ordered work packages

| ID | Deliverable | Prerequisites | Primary ownership and safe parallel work |
| --- | --- | --- | --- |
| DOC-01 | Complete delivery qualification and revision identity | Fresh main and current failure traces | Existing docs owner; CI/publication integration serialized |
| DOC-02 | Capability, tutorial and dataset coverage catalogue | Fresh native contracts; current inventory | Docs owner with scientific owners; prepare alongside DOC-01 |
| DOC-03 | Optional marimo compatibility, first notebook and static export | DOC-02 example identities; native environment | Docs owner; root dependency/CI edits through integrator |
| DOC-04 | Public-data admission and beginner/intermediate lessons | DOC-02 schema; dataset rights/semantics | Data/editorial work can run alongside DOC-03 and DOC-05 |
| DOC-05 | Astro learning components, figures and discovery | DOC-02 shared metadata contract | Docs owner; component work can run alongside notebook/data work |
| DOC-06 | Advanced curriculum across the scientific surface | Relevant DOC-02 contracts and accepted native operations | Bounded model groups; shared scientific sources remain owner-held |
| DOC-07 | R/Julia first results and migration cookbook | DOC-02 contracts; current installed-language examples | Language content can proceed alongside DOC-04/06 |
| DOC-08 | Reader/scientific/accessibility evaluation and publication | Qualified increments from DOC-01–07 | Begin reader trials with first completed lessons; final integration serial |

DOC-01 completion is a publication prerequisite, not a reason to block independent
editorial preparation. Each packet records base/head, allowed files, dependencies,
commands and results. Finishing this planning packet does not clear downstream
scientific or dependency acceptance.

Any marimo notebook execution/export in DOC-04 or DOC-06 additionally requires
the accepted DOC-03 compatibility result. Dataset acquisition, scientific/editorial
preparation, ordinary examples and Astro presentation can proceed independently.

### DOC-01 — Delivery and reproducible documentation identity

- [ ] Inspect retained traces from run 43804385173 at the pinned base. Explain the
  two mobile resource/artifact failures and repair the cause without weakening
  the artifact-identity or failed-resource assertions.
- [ ] Add package version, source commit and build identity to the site and a
  machine-readable build record. Pin generated source links and reproducible
  checkout instructions to the documented revision; retain source-file hashes.
- [~] Reconcile current hosted successes and remaining holds at the top of this
  track without transferring a pass to another commit or rewriting old receipts.
- [ ] Through the integrator, add a bounded post-publication check of build
  identity, homepage, quickstart, generated language routes and search assets.
- [~] Review and qualify the resulting artifact at its exact revision; record
  the public artifact identity separately from local build success.

Allowed surfaces: existing documentation browser helpers, generator, installation
and identity components. Shared workflow, Pages settings and root registry edits
remain integrator-owned. A skipped deployment is not a failed live-site test.

### DOC-02 — Coverage contract, model atlas and legacy inventory

- [~] Extend the existing content/example contracts with typed tutorial IDs,
  level, prerequisites, tasks, model keys/variants, operation/quantity, languages,
  dataset IDs, source/example IDs, execution mode and qualification state.
  Derive Astro's view from those records; do not create a second status authority.
- [ ] Reconcile model × variant × operation × quantity metadata with scientific
  owners and executed positive/negative cases. In particular, resolve family
  rate flags versus supported Logistic and mechanistic Bass/Gompertz paths.
- [~] Publish a model-selection guide and consistent model cards for all sixteen
  current keys. Document equations, units, domains, inputs/options/defaults,
  diagnostics, limits, sources and executable routes.
- [ ] Split the long native guide into coherent model, inference, simulation and
  execution references while preserving existing URLs and important anchors.
- [x] Inventory every root/old notebook. Classify legacy imports and outputs;
  migrate useful lessons individually or retain explicit historical status.
  Conversion to marimo is not proof of current API or scientific compatibility.
- [x] Add drift checks for unknown IDs, missing current routes, stale example
  links and newly exposed unclassified capabilities. Review the full denominator.

Acceptance: every discovered family/operation has a declared disposition; runnable,
unsupported, historical and unqualified states are distinct. Every current
tutorial resolves to checked source and an understandable learning outcome.
Ownership: docs contracts/generator plus model-owner metadata handoffs.

### DOC-03 — Native marimo compatibility and notebook publication

- [x] Test marimo 0.25.2 against actual Python 3.15.2, Polars 2.1.0,
  the locked native Innovate package and the selected platform. Record resolved
  dependencies, imported paths, native artifact identity and real execution.
- [x] Keep notebook tooling in a separately locked optional `notebooks/` project,
  after the compatibility result. ADR 0008 records this bounded decision; shared
  tech-stack reconciliation remains with the integrator. Default dependencies
  and root CI are unchanged.
- [~] Deliver the first notebook with deterministic defaults, fresh native requests
  and interpretation. The implemented public-data comparison covers Logistic and
  Gompertz; original N01 unit/rejection controls remain a later lesson.
- [~] Qualify notebook lint, explicit test collection, clean script execution,
  reactive UI changes/reset and static export. A nonzero export status fails even
  if HTML exists; known-error and hidden-state controls must demonstrate detection.
- [x] Define deterministic headless inputs through the same request-building path
  as interactive mode. Required native computations must run without a form/button
  submission; assert that results exist and match the interactive default payload.
- [x] Register the ordinary example in the existing content contract and notebook
  sources, expected parity, fixture/native/environment hashes, execution mode,
  and export identity in its linked notebook contract and receipt. The source-bound
  checker enforces a 60-second lint timeout and a 120-second export timeout;
  these limits are currently code policy, not separate manifest fields.
- [~] Publish a labelled static preview, full source and local interactive command
  through Astro. Test base-path assets, keyboard/mobile rendering, search fallback
  and existing budgets. Do not load the notebook runtime across ordinary pages.
- [x] Review the first public-data notebook's source and interpretation before
  extending the notebook curriculum. Browser interaction remains separately open.

Acceptance: a clean supported installation runs the exact source and produces the
expected native result; changing a control and restoring it reproduces the
baseline. Browser-native compatibility remains unclaimed. A separate environment
must explicitly install the same pinned native artifact, not import another wheel.

### DOC-04 — Public data and the first complete analyses

- [x] Admit D01 first using a small, frozen WDI/ITU excerpt and its metadata. Record
  licence/citation, query/vintage, raw and normalized hashes, units, denominator,
  geography, missingness and transformations. Qualify acquisition separately.
- [~] Deliver B02–B05: observation preparation, small scalar fit, comparative model
  choice, chronological assessment, diagnostics/figures and a reproducible result
  bundle. Preserve the existing analytic quickstart and an independent noisy fixture.
- [ ] Admit D02 and D03 for calendar/capacity and multiseries stock/flow lessons;
  retain workbook/table definitions and deterministic conversion recipes.
- [x] Make expected failures and poor fits teachable outcomes in D01. Do not select a
  dataset/window solely to make a model appear successful or claim identified
  diffusion mechanisms from aggregate curve fitting.
- [x] Run regular examples offline against the admitted D01 fixture.
- [ ] Add a separately invoked update command comparing upstream schema, vintage
  and revisions. It must not silently replace
  expected results or promote new data without a reviewable change.
- [~] Review each lesson's scientific interpretation and reproduce from a clean
  environment before marking the public-data case complete.

Acceptance: at least three distinct admitted empirical sources demonstrate scalar,
calendar and multiseries workflows, with reusable preparation and portable output.
These are learning cases, not a blanket empirical qualification of the library.

### DOC-05 — Make fuller use of Astro and Starlight

- [~] Extend typed collections from DOC-02; render task/model/tutorial/dataset
  catalogues with cards, prerequisites, related routes and static comparison tables.
- [~] Add reusable source/output/figure presentation through existing MDX and
  Starlight components. Extend exact-source checks for any new component with
  missing/stale-binding negative cases; retain complete runnable downloads.
- [~] Use Steps, synchronized language tabs, restrained Asides/Badges and the
  existing Expressive Code frames/markers where they improve a specific lesson.
- [ ] Generate checked SVG figures and captions. Add build-time mathematical
  rendering through a separately locked, reviewed dependency increment; use
  build-time Mermaid only for useful branching/provenance diagrams.
- [~] Extend Pagefind metadata and reader-vocabulary queries while preserving
  historical exclusions. Legacy version content and its selector are removed;
  retirement notices are excluded from indexing. Keep the catalogue usable without
  client JavaScript; add small progressive filters only where useful.
- [~] Test keyboard/mobile/theme/zoom behaviour and existing size/request budgets.
  Measure any notebook/math/figure cost and review a proposed exception explicitly.
- [ ] Review static output and live artifact identities after integration.

Acceptance: the same authoritative records drive navigation, capability labels,
example metadata and related links. No extra UI framework, server adapter, hosted
search service or browser numerical engine is required for this package.

### DOC-06 — Advanced model and operation coverage

- [x] Promote the existing Kairos, forcing, network-engine and batch examples to
  focused current lessons; retain their actual model key and engine distinctions.
- [ ] Deliver I02–I09 and A01–A06 from the coverage matrix in bounded model groups.
  Admit D04 for observed contact topology with explicitly synthetic adoption.
- [ ] Teach competition, complementarity, lock-in, latent, mixture, hierarchical,
  regime and composite semantics with controlled independent examples before any
  empirical extension. Include start-state, rate, parameter-domain and solver limits.
- [ ] Add supported uncertainty/calibration, multistart/local-rank diagnostics,
  batch failures, interval/stock/flow distinctions and fitted-state replay. New
  NUTS, ODE or regime functionality needs its actual owner acceptance, not a merged
  title or a family-level support flag.
- [ ] Preserve causal estimands and assumptions in A05. Keep optional D05 outside
  the permissive default fixture bundle until its distinct terms are resolved.
- [ ] Link each current supported operation to an executed teaching example or
  API example plus explanation; record unresolved qualifications honestly.
- [ ] Review model claims and references with the scientific owner at each
  coherent numerical/inference milestone.

Acceptance: all sixteen family keys are taught with the appropriate semantics;
advanced operation coverage has an explicit numerator/denominator and qualified
examples. No supported-looking page substitutes for a missing native capability.

### DOC-07 — Language pathways and migration recipes

- [x] Strengthen language-native documentation conventions: Python PEP 257 and
  structured docstrings, compiler-checked Rust rustdoc with FFI safety/ownership,
  Julia Markdown docstrings and R roxygen/Rd alignment. Fail on duplicate sections
  or entries and invalid convention settings; verify Python runtime documentation.
- [ ] Provide short Python, Julia and R installation/first-result routes using the
  same scientific payload and expected result, with per-language installed evidence.
- [ ] Use language tabs only for equivalent qualified tasks; explain distinct
  Polars/JSON/IPC boundaries and platform packaging limits where they differ.
- [~] Add practical old-to-current fitting, prediction, backend and interchange
  recipes. State changed scientific meaning or lack of a replacement explicitly.
- [x] Remove obsolete active notebook/documentation sources and retain immutable
  Git history with checked hashes and current migration notices. Promote only
  independently reviewed current examples; review material before export.
- [~] Verify language examples, migration links and current/retired search behaviour.

Acceptance: each maintained language has a complete route to an interpreted result;
marimo remains a Python interface rather than a claim of R/Julia notebook execution.

### DOC-08 — Reader success and completion evidence

- [ ] Run observed reader tasks on the first fit/model-selection lessons, then on
  advanced and R/Julia routes. Record task outcome, time/friction and interpretation
  errors, including participants who could not complete the task.
- [~] Complete scoped specialist review of equations, observations, uncertainty
  and public-data claims with primary references; preserve unresolved findings.
  Independent agent review and worked answers for the delivered scope are recorded;
  this does not establish human specialist or exhaustive scientific acceptance.
- [ ] Extend representative search journeys and a small cross-engine smoke set;
  add focused assistive-technology/zoom review. Retain the complete Chromium gate.
- [ ] Check current external citation/dataset links through a separate bounded
  maintenance command, distinguishing reachability from scientific support.
- [~] Produce a coverage/qualification report derived from the shared inventory:
  source structure, example execution, empirical data, model/operation coverage,
  language execution, browser, scientific review and human tasks separately.
- [ ] Verify approved publication at the public build identity and reconcile this
  track through the existing coordinator. Preserve historical evidence and holds.

## Completion and rollback

Full extended completion requires declared coverage, executed examples/notebooks,
admitted data, source-bound outputs, scientific review, reader outcomes and a
verified published artifact. A bounded package may finish while other packages
remain pending; report that scope without a whole-programme completion claim.

If a notebook or new component fails, retain the last qualified static lesson and
its runnable source. Revert the bounded change through normal version control;
do not remove historical receipts, weaken failures or introduce a fallback engine.
Package publication, new notebook hosting and browser-native ports are separate
actions, not automatic consequences of this plan.

## Historical planning verification

Research covered current repository source, documentation contracts, public site,
hosted logs, official marimo/Astro/Starlight documentation and publisher metadata.
Independent agents supplied separate notebook, Astro and curriculum/dataset reviews.
No marimo runtime or empirical analysis was executed in this planning increment.

Planning validation completed before publication:

- `git diff --check`: passed.
- JSON metadata and changed-file checks: valid `in_progress` metadata; exactly
  eight changed/added files, all inside this track.
- Relative Markdown links: all 31 checked targets resolved at the inspected base.
- Coverage cross-check: sixteen exact native model keys, 21 unique proposed lesson
  IDs, eight pending work-package markers and the recorded content/example counts.
- Independent read-only reviews: ownership/evidence, marimo design and
  curriculum/dataset semantics; resulting clarifications incorporated.
- `python scripts/conductor_registry_hygiene.py --json`: exit 1, matching the
  pre-edit failure for three completed tracks still linked under active paths
  (ODE, NUTS and CI supply-chain). No missing links or new registry findings.

The concurrent #656 reconciliation proposes changes to those registry paths;
this packet leaves that shared work with its owner. No whole native build,
notebook compatibility execution or empirical scientific acceptance is claimed.
The publication handoff records the actual remote commit and PR.
