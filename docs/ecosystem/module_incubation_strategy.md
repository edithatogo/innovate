# Ecosystem Module Incubation Strategy

## Purpose

`innovate` should be the health-intervention adoption, diffusion,
implementation-spread, and policy-diffusion package in a health economics and
outcomes research (HEOR) ecosystem. It should not own health-economic
simulation, VOI methods, generic surrogate modelling, or workflow orchestration.
Those concerns should remain in sibling modules and connect through versioned
artifacts, schemas, and adapters.

The ecosystem scope is HEOR: cost-effectiveness analysis, HTA, reimbursement,
implementation uncertainty, evidence synthesis, outcomes modelling, and
health-policy evaluation. Generic innovation modelling is not the gap this
ecosystem is intended to fill unless it directly informs health interventions,
health-policy adoption, or implementation outcomes.

## Current Ecosystem Roles

- `innovate`: health-intervention adoption curves, diffusion dynamics, policy
  spread, network adoption, substitution, competition, implementation uptake,
  and related diagnostics.
- `lifecourse`: health-economic simulation, intervention scenarios, run
  bundles, and reporting outputs.
- `voiage`: VOI analysis over uncertainty, including EVPI, EVPPI, EVSI, ENBS,
  VOI metamodels, and VOI diagnostics.
- `mars`: fixed-API MARS surrogate/metamodel package that may be used through
  public APIs only.
- HEOML: portable health-economic artifact profile and extension namespace
  model for cross-project bundles.

## Non-Goals

This incubation track does not aim to make `innovate` the owner of:

- health-economic simulation engines
- VOI methods or VOI report generation
- generic surrogate or response-surface infrastructure
- workflow orchestration across projects
- private sibling-project internals

These concerns stay in sibling modules or external contracts and are expected
to connect through public APIs, versioned schemas, and portable artifacts.

## Candidate Ecosystem Integrations

- `lifecourse` scenario adapter: provide health-intervention adoption or
  implementation uptake trajectories for health-economic scenarios.
- `voiage` uncertainty adapter: expose health-intervention adoption/diffusion
  uncertainty as a decision-relevant uncertainty source for HEOR VOI workflows.
- `mars` surrogate adapter: use MARS-style response surfaces for fitting,
  scenario approximation, or sensitivity workflows where benchmarks justify it.
- `treeage`-style operational modeling adapter: support decision-tree and
  state-transition health-economic models used in HTA and reimbursement
  workflows.
- `des` operational modeling adapter: support discrete-event simulation for
  patient pathways, queueing, resource utilization, and pathway-level timing
  uncertainty.
- HEOML `innovate` extension: define uptake, adoption, diffusion, policy-spread,
  and network trace artifacts for health-economic run bundles.
- Future policy/reporting tools: consume HEOR adoption artifacts without
  importing `innovate` internals.

## Artifact Direction

`innovate` HEOR ecosystem artifacts should be portable and binding-friendly:

- JSON for manifests, small metadata, diagnostics, and provenance.
- Arrow or Parquet for adoption curves, parameter samples, model predictions,
  network traces, and scenario outputs.
- CSV for simple review exports.
- JSON Schema for compatibility validation when artifacts cross repository
  boundaries.

Pickle is not a portable ecosystem contract.

### Core Artifact Shapes

- `adoption_curve`: `scenario_id`, `intervention_id`, `time`, `adoption`,
  `cumulative_adoption`, `population`, `segment`, `uncertainty_label`.
- `uptake_trajectory`: `scenario_id`, `arm_id`, `time`, `uptake_rate`,
  `reach`, `adherence`, `retention`, `uncertainty_label`.
- `policy_spread_trace`: `policy_id`, `network_id`, `time`, `exposure`,
  `adoption`, `coverage`, `segment`, `uncertainty_label`.
- `network_diffusion_trace`: `network_id`, `node_id`, `edge_id`, `time`,
  `activation`, `influence`, `peer_effect`, `uncertainty_label`.
- `event_trace`: `simulation_id`, `entity_id`, `event_time`, `event_type`,
  `state_before`, `state_after`, `resource_id`, `queue_time`.
- `simulation_run_bundle`: `model_id`, `scenario_id`, `run_id`, `seed`,
  `inputs_uri`, `outputs_uri`, `event_log_uri`, `status`.
- `diagnostics_record`: `model_id`, `fit_metric`, `calibration_target`,
  `residual_summary`, `convergence_status`, `package_version`, `created_at`.
- `provenance_record`: `schema_version`, `source_model`, `source_commit`,
  `software_version`, `seed`, `generated_at`.

Tabular artifacts should default to Arrow or Parquet so sibling modules can
consume them without Python object coupling.

### Operational Modeling Fixture Contracts

The operational-modeling fixture contracts are documented fixtures, not runtime
engine integrations:

- [operational_modeling/treeage_style/manifest.json](../../specs/ecosystem/operational_modeling/treeage_style/manifest.json)
  defines a TreeAge-style decision-tree and state-transition contract for HTA
  and reimbursement examples. It includes strategy, state, transition, payoff,
  schema-version, provenance, and XLA eligibility or rejection fields while
  keeping proprietary TreeAge parsing out of scope.
- [operational_modeling/des/manifest.json](../../specs/ecosystem/operational_modeling/des/manifest.json)
  defines a DES contract around event-log rows, queue metrics, resource
  identifiers, pathway states, deterministic ordering, run metadata,
  provenance, and XLA eligibility or rejection notes. It represents pathways as
  artifacts rather than private engine state.

These fixtures are the documented stage of the adapter promotion ladder.
Runtime simulation engines out of the current `innovate` package remain out of
scope unless a future adapter is isolated behind optional extras and passes the
same smoke CI, Renovate, security, compatibility-matrix, documentation, and
removal-path gates as other ecosystem integrations.

TreeAge-style state-transition calculations may be XLA-eligible when they are
bounded matrix operations with static cycle counts. Classic DES is rejected for
XLA when dynamic event queues, resource contention, or runtime-created events
would be distorted; a separate vectorized Monte Carlo approximation can be
evaluated later only if it preserves the documented artifact semantics.

## HEOML Alignment

The future `heoml.extensions.innovate` namespace should cover:

- health-intervention and implementation metadata
- adoption and uptake trajectories
- policy diffusion traces
- network diffusion summaries
- decision-tree and state-transition operational modeling metadata
- discrete-event simulation pathways, queueing, and event logs
- uncertainty and parameter-draw metadata
- calibration and fit diagnostics
- provenance and software-version metadata

The HEOML extension should reference stable `innovate` artifact schemas and
public functional-kernel semantics, not private implementation classes.

### Schema Placement

The interim HEOML extension schema home is
`specs/ecosystem/heoml/extensions/innovate/`. This repo-local placement keeps
the `heoml.extensions.innovate` contract close to the `innovate` artifact
fixtures and Arrow-compatible interchange checks while HEOML is not yet a
standalone `heoml` repository.

The placement decision is recorded in
[ADR 0005: HEOML Schema Placement](../adr/0005-heoml-schema-placement.md).
The migration trigger is a standalone `heoml` repository with a published
semver schema bundle, stable extension namespace, cross-repository fixture CI,
and a documented deprecation window for the repo-local schemas.

HEOML extension contracts must use `schema_version` fields, binding-friendly
JSON manifests, JSON Schema validation, and Arrow-compatible tabular payloads.
They MUST NOT use private Python objects, MUST NOT use pickle, and MUST NOT use
private Python object framing.

### Boundary Rule

- Generic `innovate` artifacts are the portable runtime outputs for adoption,
  diffusion, and diagnostics.
- HEOML artifacts are wrappers or namespace mappings around those portable
  outputs when a health-economic bundle needs a shared extension profile.
- If an artifact cannot be represented without a private implementation object,
  it does not belong in the ecosystem contract.

### Namespace Rule

- Use plain `innovate` artifact names for local workflows, bindings, and
  repository-internal interchange.
- Use `heoml.extensions.innovate.*` only when a sibling project or downstream
  HEOR bundle needs a cross-repo, schema-versioned extension surface.
- Keep the underlying tabular payload identical across both forms so wrappers
  stay thin and mechanical.

## Dependency Policy

- Keep the base `innovate` install independent of `lifecourse`, `voiage`, and
  `mars`.
- Reject or defer integrations that are useful for generic diffusion modelling
  but do not materially support HEOR workflows.
- Add ecosystem integrations through optional extras only after stable public
  APIs and fixture contracts exist.
- Require smoke CI, Renovate coverage, security checks, documentation, and a
  removal path for each optional adapter dependency.
- Treat every optional adapter as disabled by default and load it only through
  an explicit extra, plugin, or adapter flag.
- Require a deterministic smoke fixture for each adapter before it can be
  treated as more than a documented concept.
- Require a compatibility matrix that names supported `innovate`, adapter, and
  sibling-package versions before an integration is called supported.
- Do not require changes to the `mars` core API. Add adapter logic in
  `innovate` or a future companion package if needed.

## Promotion Criteria

1. Document artifact contracts and dependency policy.
2. Add deterministic compatibility fixtures.
3. Add experimental adapters behind optional extras.
4. Add cross-repo smoke CI and fixture validation.
5. Promote only after version compatibility, docs, release notes, and
   deprecation policy are clear.

### Promotion Stages

- Documented: the integration is specified, but no runtime path ships yet.
- Experimental: the adapter exists behind an optional extra and is exercised
  by smoke CI and fixtures, but it may change without deprecation guarantees.
- Supported: the adapter has version compatibility matrices, release notes,
  deprecation policy, and an explicit removal path.

### Required Gates

- Optional extras must be explicit and narrow in scope.
- Smoke CI must cover adapter import, fixture loading, and one end-to-end
  representative call path.
- Renovate or equivalent dependency automation must monitor the adapter
  dependency set.
- Security checks must cover the adapter package and its transitive runtime
  dependencies.
- Documentation must state the contract, the supported versions, and the
  removal path.
- Supported adapters must declare a compatibility matrix before promotion.

## Immediate Follow-Up

- Define a minimal adoption-trajectory fixture that `lifecourse` can consume:
  `specs/ecosystem/lifecourse/adoption_trajectory/v1/manifest.json` documents
  the first deterministic Parquet smoke fixture. This is a documented contract
  only; runtime adapter implementation remains future work until optional
  extras, smoke CI, and a compatibility matrix are in place.
- Define a diffusion-uncertainty fixture that `voiage` can use for VOI examples:
  `specs/ecosystem/voiage/uncertainty/diffusion_v1/manifest.json` documents
  parameter draws, adoption trajectories, provenance, sample dimensions, and
  VOI concept mappings. VOI method implementation remains owned outside `innovate`.
  This fixture is a decision-relevant uncertainty source, not an EVPI, EVPPI,
  EVSI, ENBS, or reporting engine.
- Define a TreeAge-style operational modeling fixture for reimbursement and
  decision-analysis workflows.
- Define a DES fixture with event logs and queue metrics for pathway timing
  examples.
- Decide whether HEOML extension schemas should live in `innovate`, `lifecourse`
  while HEOML is embedded, or a future standalone `heoml` repository.
- Benchmark whether `mars` improves adoption-curve surrogate workflows before
  exposing it as a supported optional backend.
- HEOR naming brainstorm: shortlist `calibrate`, `evidence`, `process`,
  `report`, `registry`, `workflow`, `quality`, `engines`, and `heoml`; keep
  PM4Py in the ecosystem-only process-mining bucket; require a CLI surface for
  every future module and an explicit MCP decision where orchestration matters.
- Define a documented-stage process-mining fixture:
  `specs/ecosystem/process/fixtures/event_log_v1/manifest.json` records a
  portable synthetic event-log bundle with pathway discovery, conformance, and
  bottleneck summary payloads. CLI support is planned before adapter
  implementation, MCP remains deferred, and PM4Py stays a reference candidate
  rather than a base dependency.

## HEOR Naming Brainstorm

The current HEOR naming shortlist is documented as a planning target only, not
an implementation commitment. The reserved names are:

- `calibrate`
- `evidence`
- `process`
- `report`
- `registry`
- `workflow`
- `quality`
- `engines`
- `heoml`

PM4Py remains in the ecosystem-only process-mining bucket. Any future module
that comes out of this list should expose a CLI surface, and any MCP interface
decision should be explicit rather than assumed.

The archived track for this work is
[HEOR Module Naming Brainstorm](../../conductor/archive/heor_module_naming_brainstorm_20260429/).

## MARS Surrogate Benchmark Gate

The MARS surrogate benchmark gate is the evidence path for deciding whether
`mars` should become an optional backend for adoption-curve surrogate workflows.
The current outcome is **defer**. `mars` remains outside base and optional
package metadata until opt-in benchmark evidence records reference
NumPy/SciPy behavior, MARS surrogate behavior, eligible XLA-backed alternatives,
dependency cost, failure modes, and gain attribution.

Fast CI should validate only the benchmark-gate metadata. Timing evidence should
be generated through the opt-in command documented in the benchmark workflow
tutorial, and any promotion decision should keep surrogate gains separate from
JAX/XLA compile and steady-state runtime effects.
