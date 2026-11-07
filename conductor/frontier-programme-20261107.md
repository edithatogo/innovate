# Scientific frontier programme — 7 November 2026

## Status and authority

The user authorised recording the recommendations in Conductor. This is a planning change, not implementation, native qualification, a completed study, or permission to publish packages or submit manuscripts. All new implementation tasks remain open.

Source baseline: `b9998addf32717438df92b23a306903e4db51620`. Inspection found a hybrid Python/Rust library, not a fully Rust-owned scientific core. Historical completed tracks remain historical records; their titles are not evidence that all current operations, payloads and model families are native or statistically validated.

The existing [scientific ecosystem alignment track](tracks/scientific_ecosystem_alignment_20261029/index.md) retains its owner, accepted pins, installed-consumer evidence and unfinished gates. Reuse its [2 November reconciliation](tracks/scientific_ecosystem_alignment_20261029/reconciliation-20261102.md). Reconcile overlapping open PRs before implementing; do not silently duplicate or supersede them.

## Objective and alternatives

Reach the frontier through scientific correctness, a genuine native core and independent comparative validation. Investigate beyond-frontier contributions through adoption–capacity coupling and joint implementation/research decisions. Novelty and superiority remain hypotheses to test.

Retaining Python as the authoritative core conflicts with the requested architecture. A wholesale rewrite would discard useful reference behaviour and expand risk. The selected direction is incremental inversion: extract validated Rust model-operation slices, make language bindings call them, and retain independent reference implementations as test oracles or explicitly named external backends.

## Registered work

| Track | Priority | Dependency and independently useful outcome |
|---|---|---|
| [Scientific integrity](tracks/scientific_integrity_20261107/index.md) | P0 | Repair invalid uncertainty/causal semantics; no dependency on a full rewrite. |
| [Rust-core convergence](tracks/rust_core_convergence_20261107/index.md) | P0 | Extract the native core; promotion requires integrity tests and existing alignment gates. |
| [Empirical validation and publication](tracks/empirical_validation_publication_20261107/index.md) | P1 | Acquire/version data now; publish numerical claims only after relevant correctness gates. |
| [Advanced diffusion methods](tracks/advanced_diffusion_frontier_20261107/index.md) | P2 | Promote one independently validated model/likelihood/inference slice at a time. |
| [Ecosystem decision integration](tracks/ecosystem_decision_integration_20261107/index.md) | P2 | Preserve joint uncertainty across Innovate, Kairos, economic models and Voiage. |
| [Healthcare hype evaluation](tracks/healthcare_hype_validation_20261107/index.md) | P1 research | Begin source/construct audits now; fitted scientific claims require the appropriate integrity and model gates. |

## Shared contracts and ownership

Innovate owns adoption dynamics, observation likelihoods, estimation and diagnostics. Kairos owns simulation time, event ordering, agent/resource state and simulation random streams. Voiage owns value-of-information and decision-uncertainty calculations. Application repositories own clinical pathways, costs, utilities and valuation. MARS is an optional surrogate, promoted only after accuracy/cost and decision-preservation evidence.

Align with `edithatogo/repository-standards` and Voiage's native separation, but inspect and pin the relevant versions before implementation. Do not assume another repository has completed its own native migration. `careops-sim`, `tobac` and `vop_poc_nz` are candidate application consumers, not dependencies of the generic numerical core.

Typed interchange must retain units, denominators, observation type, time basis, missingness, censoring, model/data/software versions, draw/scenario/strategy/product/region/time identifiers, fit status and provenance. Preserve correlated joint draws; do not independently resample marginal intervals. Arrow is an interchange boundary, not a mandate for Arrow tables inside every numerical operation. Dataframe and accelerator adapters remain optional.

## Evidence and release rules

1. Distinguish verification, predictive validation, causal identification and decision utility. A passing schema test does not establish any of the latter three.
2. A native claim requires execution without Python installed, runtime-dispatch evidence and installed consumers. Unsupported native requests fail explicitly; no undisclosed subprocess fallback.
3. Inference failure cannot produce posterior-labelled draws. Unsupported estimators cannot silently return unadjusted contrasts.
4. Freeze tolerances, comparison methods, data splits and computational budgets before inspecting final benchmark results. Separate exploratory tuning from confirmation.
5. Record failures, exclusions, resource consumption and negative findings. Native parity tests use an independent mathematical oracle, not fixtures generated solely by the same implementation.
6. Generate capability/claim summaries from executable evidence. Do not close an implementation track because plans, manifests or documentation exist.

## Agent delivery and integration

The integrator owns shared schemas, dependency pins and root manifests. Independent workers may implement bounded model slices, source adapters and benchmark runners on separate branches after reading both spec and plan. Every implementation task follows failing regression, minimal implementation, review, focused verification and installed-consumer acceptance. Integration proceeds by evidence, not calendar promises; human evidence annotation and external review are not accelerated by assigning more coding agents.

Before merging a slice, run its focused tests plus the repository's required gates. Use `python scripts/conductor_registry_hygiene.py --json` for Conductor structure. Archive only after numerical, native, consumer and documentation gates relevant to the claim pass. Do not waive existing native/HPC gates or imply scheduler/cluster acceptance from local CPU tests.

## Research outputs

The [validation matrix](research/validation-matrix-20261107.md) maps model/features to candidate data and tests. The [healthcare hype protocol](research/healthcare-hype-protocol-20261107.md) defines a separate, falsifiable study.

Target JOSS first for the delivered software scope and real research use; International Journal of Forecasting for comparative predictive evaluation; Technological Forecasting and Social Change for a substantive diffusion/policy contribution; Scientific Data for a reusable benchmark; JASSS for network/agent methods; Medical Decision Making for adoption–capacity–VOI decisions. For the hype measurement study, prefer JMIR Infodemiology, with JMIR for a broader validated contribution. These are targets, not acceptance forecasts.

Check each journal's current requirements at submission. JOSS requires actual research use and AI-assistance disclosure and restricts AI-mediated author–editor/reviewer conversations. Authors must handle those interactions themselves in accordance with policy. Do not submit or claim independent human review through an agent.

No paid compute, private clinical data ingestion, credential changes, external repository mutations, package publication or manuscript submission is authorised by this planning update.
