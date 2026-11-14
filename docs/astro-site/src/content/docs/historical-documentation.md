---
title: Historical documentation and current routes
description: Find current guidance for every retained pre-0.7 page, with explicit partial replacements and gaps.
---

Use this map when an earlier link brings you to a page marked **Pre-0.7 reference**.
The 61 pages below remain at their existing URLs so that earlier analyses and
links can be understood. Their examples, package names, backend descriptions and
release receipts belong to the earlier implementation.

For new work, start with the [current tutorials](/innovate/tutorials/),
[native capabilities and limits](/innovate/native-migration/) or
[Python, Julia and R references](/innovate/bindings/). The separate
[0.6 reference archive](/innovate/latest/) is retained unchanged.

## How to read the map

The status describes a navigation review of the stated reader task. It does not
qualify the historical page, prove scientific correctness or establish feature
parity. Read each row's scope before moving an analysis to a current interface.

| Status | Meaning |
| --- | --- |
| Current replacement | A current route addresses the named entry task; its own limits still apply. |
| Partial replacement | Current guidance addresses part of the task, with the missing or different scope stated explicitly. |
| Unsupported interface | The earlier interface or backend is outside the maintained native API. A related current capability may use different scientific semantics. |
| No qualified replacement | This review has no current replacement for the stated policy or workflow. A related reference is context, not proof that the task has been migrated. |

The map contains 8 current replacements, 37 partial replacements, 3 unsupported
interfaces and 13 entries without a qualified replacement. All 61 source pages
retain their historical classification. A replacement link never converts an
old readiness flag or successful run into current evidence.

## Architecture, bindings and diagnostics

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [ADR](/innovate/architecture/adr/) | No qualified replacement | The earlier page is a pointer to decision history. The [native kernel contract](/innovate/core/kernel/) describes today's execution boundary; it is not a replacement ADR catalogue or a review of those earlier decisions. |
| [Architecture](/innovate/architecture/) | Partial replacement | Start with [core concepts](/innovate/core/) and [native bindings](/innovate/bindings/) for current navigation. The earlier architecture collection has not been requalified as a whole. |
| [Polyglot Repository Architecture](/innovate/architecture/polyglot-repo/) | Partial replacement | The [native kernel](/innovate/core/kernel/) and [Rust core](/innovate/bindings/rust/) identify current scientific ownership. The old Python-reference ownership map and seven-language publication layout do not describe the maintained binding scope. |
| [Julia Bindings](/innovate/bindings/julia/) | Current replacement | Use the [generated Julia reference](/innovate/api/generated/julia/) and [native binding overview](/innovate/bindings/) for C ABI calls and library discovery. The earlier Python subprocess bridge and registry-installation claims do not transfer to the current package. |
| [Binding parity](/innovate/bindings/parity/) | Partial replacement | The [binding overview](/innovate/bindings/) names Python, Julia and R; [documentation contracts](/innovate/maintainers/documentation-quality/) separate source documentation from language execution evidence. Neither is a claim of complete cross-language scientific or platform parity. |
| [Diagnostics Contract](/innovate/core/diagnostics-contract/) | Partial replacement | Read [fitting](/innovate/user-guide/fitting/) and [FitResult](/innovate/api/generated/fit-result/) for the returned envelope and model-dependent diagnostics. The historical common artifact shape is not guaranteed by the current result type. |

## Maintainer policies

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [Binding Conformance CI](/innovate/maintainers/binding-conformance/) | Partial replacement | [Documentation contracts](/innovate/maintainers/documentation-quality/) describe current Python, Julia and R source checks and execution receipts. Those gates do not replace complete binding conformance, package checks or scientific parity evidence. |
| [Compatibility Policy](/innovate/maintainers/compatibility/) | Partial replacement | Use the [native kernel contract](/innovate/core/kernel/) for schema-2.0 behavior and [native bindings](/innovate/bindings/) for supported languages. Cross-version artifacts and platform distribution still require their own evidence. |
| [Deployment Readiness](/innovate/maintainers/deployment-readiness/) | Current replacement | Use [documentation quality contracts](/innovate/maintainers/documentation-quality/) for the current build, artifact checks, publication boundary and recovery guidance. A retained deployment receipt cannot establish the state of a new build or live deployment. |
| [Deprecation Policy](/innovate/maintainers/deprecation/) | No qualified replacement | The [native migration guide](/innovate/native-migration/) describes breaking 0.7 changes and capability limits. It does not establish a replacement deprecation-window policy for every earlier public interface. |
| [DocSearch Gate](/innovate/maintainers/docsearch/) | Partial replacement | [Documentation contracts](/innovate/maintainers/documentation-quality/) cover built search artifacts. The earlier Algolia status is not a current provider-verification receipt; external indexing and deployment configuration need separate checks. |
| [Maintainers](/innovate/maintainers/) | Partial replacement | Use [documentation contracts](/innovate/maintainers/documentation-quality/) and [local test acceleration](/innovate/maintainers/test-acceleration/) for their current maintainer tasks. The remaining policies in the earlier collection are still historical. |
| [Maintenance Policy](/innovate/maintainers/maintenance/) | No qualified replacement | No current replacement for the earlier maintenance cadence has been reviewed here. [Documentation contracts](/innovate/maintainers/documentation-quality/) explain run-specific evidence; they do not establish a repository-wide maintenance commitment. |
| [Package Health](/innovate/maintainers/package-health/) | Partial replacement | The [native migration guide](/innovate/native-migration/) identifies current capability and release limits; [research registry readiness](/innovate/operations/registry-readiness/) covers its named destinations. These are not a live package-health dashboard. |
| [Plugins Module](/innovate/maintainers/plugins/) | Partial replacement | Use [invoke_registered_extension](/innovate/api/generated/invoke-registered-extension/) for trusted callback resolution and output validation. The old module overview does not define the current callback contract or establish sandboxing. |
| [Publication](/innovate/maintainers/publication/) | No qualified replacement | The [binding overview](/innovate/bindings/) establishes current language scope. No complete replacement for the earlier multi-registry publication procedure is qualified here; historical package names, bridge assets and version gates are not current installation instructions. |
| [Release Notes](/innovate/maintainers/release-notes/) | No qualified replacement | No migrated release-automation policy is qualified by this map. Consult the [native migration guide](/innovate/native-migration/) for current API changes; do not reuse the earlier package-version matrix as a current release contract. |
| [Release Readiness](/innovate/maintainers/release-readiness/) | Partial replacement | Read the [native migration guide](/innovate/native-migration/) for open scientific and distribution obligations and [documentation contracts](/innovate/maintainers/documentation-quality/) for documentation gates. Passing those documentation gates alone does not declare a release ready. |
| [Runtime Logging](/innovate/maintainers/runtime-logging/) | Partial replacement | The [native kernel contract](/innovate/core/kernel/) and [NativeError](/innovate/api/generated/native-error/) describe current failure information. They do not establish the earlier project-wide logging and instrumentation policy. |
| [Stability Module](/innovate/maintainers/stability/) | No qualified replacement | The current [extension callback reference](/innovate/api/generated/invoke-registered-extension/) describes an invocation boundary. It is not a replacement for the historical stability-tier module or a complete lifecycle guarantee. |
| [Support Policy](/innovate/maintainers/support/) | No qualified replacement | Use [native bindings](/innovate/bindings/) to identify maintained language interfaces. The earlier support matrix does not establish a current response-time, funding or service commitment. |

## Documentation migration history

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [Archive](/innovate/migration/archive/) | No qualified replacement | The [0.6 reference](/innovate/latest/) remains available as versioned documentation. This map does not replace a reviewed catalogue of every Conductor archive or requalify historical release and registry evidence. |
| [Migration](/innovate/migration/) | Current replacement | Use [native migration](/innovate/native-migration/) for current API migration and [documentation contracts](/innovate/maintainers/documentation-quality/) for active Astro checks. The retained page records the earlier documentation cutover. |
| [Redirects](/innovate/migration/redirects/) | Partial replacement | This page's tables cover every retained current-path historical page. They do not verify every former Sphinx redirect; [documentation contracts](/innovate/maintainers/documentation-quality/) explain rendered route and link checks. |
| [Migration References](/innovate/migration/references/) | Partial replacement | Use [native migration](/innovate/native-migration/) and [documentation contracts](/innovate/maintainers/documentation-quality/) as current entry points. Earlier architecture, registry and archive references remain historical evidence. |
| [Validation](/innovate/migration/validation/) | Current replacement | [Documentation contracts](/innovate/maintainers/documentation-quality/) give current source, execution and artifact checks. A previous successful cutover or build is not evidence that the current commit passes those checks. |

## Operations and release planning

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [ABI and Binary Compatibility](/innovate/operations/abi-compatibility/) | Partial replacement | Read the [kernel contract](/innovate/core/kernel/), [Arrow interchange](/innovate/core/arrow-interchange/) and [binding overview](/innovate/bindings/) for today's JSON/C ABI and copied IPC boundaries. They do not promise the earlier proposed C Data Interface, zero-copy path or all-platform binary compatibility. |
| [Accelerator and Parallel Execution Evidence](/innovate/operations/accelerator-parallel-execution/) | Partial replacement | [Runtime choices](/innovate/user-guide/backends/) and [native execution modes](/innovate/native-migration/#execution-modes) describe local CPU workers and bounded benchmark sharding. Accelerator, multi-node and live scheduler qualification remain separate from those checks. |
| [Community Submission Readiness](/innovate/operations/community-readiness/) | Partial replacement | [Research registry readiness](/innovate/operations/registry-readiness/) covers Software Heritage, RRID and JOSS evidence. It does not establish current readiness or acceptance for every community and foundation in the earlier matrix. |
| [Governance and Sustainability](/innovate/operations/governance/) | No qualified replacement | No current replacement governance-and-sustainability dossier is qualified here. Earlier maintainer, funding, sponsor and foundation statements require current evidence before they are reused. |
| [HPC Packaging and Registry Readiness](/innovate/operations/hpc-readiness/) | No qualified replacement | Use [development installation](/innovate/user-guide/installation/) for the current package and [native execution modes](/innovate/native-migration/#execution-modes) for bounded runner behavior. These do not qualify the earlier Spack, EasyBuild, HPSF or E4S recipes and readiness states. |
| [HPC Registry Contract](/innovate/operations/hpc-registry/) | No qualified replacement | No current native HPC-registry submission contract is qualified by this map. [Native execution modes](/innovate/native-migration/#execution-modes) describe execution limits, not upstream registry readiness or acceptance. |
| [HPC Submission Packet](/innovate/operations/hpc-submission-packet/) | No qualified replacement | The earlier submission packet and readiness flags have no qualified current replacement here. A fresh packet must identify the current native build, target requirements and actual evidence; [native migration](/innovate/native-migration/) records capability limits. |
| [HPC Submission Workflow](/innovate/operations/hpc-submission-workflow/) | No qualified replacement | No current end-to-end HPC submission workflow is qualified here. The [native execution guide](/innovate/native-migration/#execution-modes) is related runner guidance and does not validate the old scheduler submission commands or upstream handoffs. |
| [Operations](/innovate/operations/) | Partial replacement | Use [native capabilities](/innovate/native-migration/) for runtime behavior and [research registry readiness](/innovate/operations/registry-readiness/) for its named destinations. The retained operations collection is not a current maturity dashboard. |
| [Polyglot Registry Plan](/innovate/operations/polyglot-registry/) | Partial replacement | [Native bindings](/innovate/bindings/) replaces the language-scope entry point, and [research registry readiness](/innovate/operations/registry-readiness/) covers three research destinations. The wider package-manager and HPC registry plan remains unqualified here. |
| [Registry Submission Receipts](/innovate/operations/registry-submissions/) | Partial replacement | Use [research registry readiness](/innovate/operations/registry-readiness/) for current recorded research-destination status. Historical package and registry receipts do not prove that the 0.7 native artifacts have been published or accepted. |
| [Release Maturity Dashboard](/innovate/operations/release-maturity/) | Partial replacement | The [native migration guide](/innovate/native-migration/) and [documentation contracts](/innovate/maintainers/documentation-quality/) state their current obligations. They do not replace a complete release-maturity dashboard or establish external acceptance. |
| [Remote Execution](/innovate/operations/remote-execution/) | Partial replacement | The [native remote-worker guide](/innovate/native-migration/#run-a-native-remote-worker-subprocess) documents a single-request subprocess and nested error checks. Hosted authentication, tenant isolation and transport qualification remain separate; the older in-process Python adapter is not the current interface. |
| [Roadmap](/innovate/operations/roadmap/) | Current replacement | The [native migration guide](/innovate/native-migration/) is the current entry point for implemented native capabilities and remaining obligations. Earlier Python-bridge ownership and archived completion claims do not describe the 0.7 execution model. |
| [Runtime Logging](/innovate/operations/runtime-logging/) | Partial replacement | Use the [kernel contract](/innovate/core/kernel/) for current structured failures and the [remote-worker guide](/innovate/native-migration/#run-a-native-remote-worker-subprocess) for process diagnostics. These do not qualify the earlier universal logging schema or release-evidence pipeline. |
| [Rust Core](/innovate/operations/rust-core/) | Current replacement | Use [Rust core](/innovate/bindings/rust/), the [native kernel](/innovate/core/kernel/) and [native capabilities](/innovate/native-migration/). Rust owns current scientific execution; the earlier selective promotion and Python numerical fallback are historical. |
| [Scientific and HPC Readiness Roadmap](/innovate/operations/scientific-hpc/) | Partial replacement | [Native execution modes](/innovate/native-migration/#execution-modes) and [research registry readiness](/innovate/operations/registry-readiness/) provide narrower current evidence. They do not qualify the earlier complete scientific/HPC roadmap, accelerator coverage or portability commitments. |
| [Submission Readiness Dossiers](/innovate/operations/submission-dossiers/) | Partial replacement | [Research registry readiness](/innovate/operations/registry-readiness/) names current Software Heritage, RRID and JOSS work. A qualified replacement for the wider CRAN, community and foundation dossier collection is still missing. |
| [XLA Backend Strategy](/innovate/operations/xla-backend/) | Unsupported interface | The earlier optional JAX/XLA and NumPy fallback strategy is outside the maintained native runtime described in [runtime choices](/innovate/user-guide/backends/). Current [execution modes](/innovate/native-migration/#execution-modes) keep accelerator qualification explicit. |

## Earlier technical roadmaps

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [DataFrame Engine Experiments](/innovate/roadmap/dataframe-engine/) | Current replacement | Use [Arrow and Polars interchange](/innovate/core/arrow-interchange/) for the current required Polars interface, collection boundary and copied IPC. The earlier pandas default and optional-Polars fallback policy no longer describe this interface. |
| [Diagnostics and Uncertainty Artifacts](/innovate/roadmap/diagnostics-uncertainty/) | Partial replacement | Read [fitting](/innovate/user-guide/fitting/), [native capabilities](/innovate/native-migration/) and the [Bayesian tutorial](/innovate/tutorials/bayesian-fitter/) for operation-specific evidence. The historical diagnostics artifact schema is not a universal current return contract. |
| [Probabilistic Inference](/innovate/roadmap/probabilistic-inference/) | Partial replacement | The [native Bayesian tutorial](/innovate/tutorials/bayesian-fitter/) and [native inference guide](/innovate/native-migration/#bayesian-inference-for-scalar-forecasts) cover bounded Gaussian-error inference. They do not restore the former NumPyro, BlackJAX or TensorFlow Probability interfaces. |

## Specialist tutorials and advanced modeling

| Retained historical page | Navigation status | Current guidance and scope |
| --- | --- | --- |
| [Advanced Diffusion Inference](/innovate/tutorials/advanced-diffusion-inference/) | Partial replacement | The [native guide](/innovate/native-migration/) describes grouped point fits, latent residual smoothing and regime switching with their separate limits. Those ports do not establish hierarchical posterior inference, general state-space inference or the historical simulation-draw behavior. |
| [Advanced runtime workflows](/innovate/tutorials/advanced-runtime/) | Partial replacement | Use [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/) for checked scenario execution and [native capabilities](/innovate/native-migration/) for other operations. These pages do not qualify the earlier complete ensemble, streaming-update and interval-calibration workflow. |
| [Benchmark Workflows and Model Cards](/innovate/tutorials/benchmark-workflows/) | Partial replacement | The [native execution guide](/innovate/native-migration/#execution-modes) documents current benchmark sharding and receipts; [runtime choices](/innovate/user-guide/backends/) explains performance-claim limits. The historical corpus, model-card and accelerator promotion workflow is not thereby requalified. |
| [Explainability, Sensitivity, and Decision Reports](/innovate/tutorials/decision-reports/) | Partial replacement | [Core concepts](/innovate/core/), [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/) and [causal contrasts](/innovate/tutorials/counterfactual-analysis/) explain result interpretation. They do not reproduce the earlier report constructors, sensitivity helpers or export workflow. |
| [Shared Diagnostics Contract and Uncertainty](/innovate/tutorials/diagnostics-contract/) | Partial replacement | Use [fitting](/innovate/user-guide/fitting/), [FitResult](/innovate/api/generated/fit-result/) and the [Bayesian tutorial](/innovate/tutorials/bayesian-fitter/). Current diagnostics depend on the operation; earlier diagnostics classes and plotting examples are not current API instructions. |
| [Functional Kernel Contract](/innovate/tutorials/functional-kernel-contract/) | Current replacement | Use the [native kernel contract](/innovate/core/kernel/) and [execute reference](/innovate/api/generated/execute/) for schema-2.0 requests, envelopes, errors and retained state. The older schema-1.0 Python object model is not the current calling interface. |
| [JAX Backend for High-Performance Computing](/innovate/tutorials/jax-backend/) | Unsupported interface | The earlier `use_backend` and `BassModel` JAX example is outside the current maintained API. Start with [runtime choices](/innovate/user-guide/backends/); native CPU execution does not establish a JAX, GPU or TPU replacement. |
| [Kairos Simulation Adapter](/innovate/tutorials/kairos-simulation-adapter/) | Partial replacement | The [native guide](/innovate/native-migration/) documents current Bass DES/ABM and network-service engines, while [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/) checks replay. These use native contracts and do not promise the old Python adapter or telemetry schema. |
| [Multi-Product Diffusion Model](/innovate/tutorials/multi-product/) | Partial replacement | Use the competition case in [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/) and [native competitive-model conventions](/innovate/native-migration/#competitive-models). They establish specified native equations and limits, not complete parity with every historical object-model helper. |
| [NDlib Integration for Network-Based Diffusion](/innovate/tutorials/ndlib-integration/) | Unsupported interface | The old NDLib adapter and `legacy-abm` installation example are outside the maintained native API. [Native network service simulation](/innovate/native-migration/#network-imitation-with-resource-service) addresses a bounded related task with different semantics; it is not a drop-in NDLib replacement. |
| [Plugin API and Stability Tiers](/innovate/tutorials/plugin-api-stability/) | Partial replacement | Use [invoke_registered_extension](/innovate/api/generated/invoke-registered-extension/) for current manifest resolution, trusted callback execution and validation. This is narrower than the earlier Python registration and lifecycle tutorial. |
| [Handling Seasonal Data with Diffusion Models](/innovate/tutorials/seasonal-data/) | Partial replacement | The [seasonal_decomposition reference](/innovate/api/generated/seasonal-decomposition/) describes the native smoother; [sarima_fit](/innovate/api/generated/sarima-fit/) returns in-sample values. Neither qualifies the earlier trend-fitting pipeline or supplies forecast uncertainty by itself. |
| [Advanced Policy, Competition, and Substitution Modeling](/innovate/user-guide/advanced-modeling/) | Partial replacement | Use [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/), [successive-generation fitting](/innovate/tutorials/norton-bass/) and [native capabilities](/innovate/native-migration/). Their bounded tasks do not establish complete intervention-targeting, equilibrium, sensitivity or threshold-helper parity. |

## Search and qualification

These current-path historical pages are excluded from the site's bundled
Pagefind index using Starlight's
[`pagefind: false` frontmatter](https://starlight.astro.build/reference/frontmatter/#pagefind).
Their routes and visible migration notices remain available through this map.
This index and the current documentation remain discoverable. The versioned
0.6 archive is preserved separately; this change does not reclassify or edit it.

An external search provider, including an Algolia deployment, has its own
indexing rules and needs separate qualification. A Pagefind exclusion is not
evidence that an external crawler has updated its index.

The [documentation quality contract](/innovate/maintainers/documentation-quality/)
continues to count retained historical pages as unresolved whole-site scope.
Search presentation does not retire that debt. Before promoting a page to current
status, its actual API instructions, scientific interpretation, executable
examples and required evidence must be reviewed and checked.
