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

## HEOML Alignment

The future `heoml.extensions.innovate` namespace should cover:

- health-intervention and implementation metadata
- adoption and uptake trajectories
- policy diffusion traces
- network diffusion summaries
- uncertainty and parameter-draw metadata
- calibration and fit diagnostics
- provenance and software-version metadata

The HEOML extension should reference stable `innovate` artifact schemas and
public functional-kernel semantics, not private implementation classes.

## Dependency Policy

- Keep the base `innovate` install independent of `lifecourse`, `voiage`, and
  `mars`.
- Reject or defer integrations that are useful for generic diffusion modelling
  but do not materially support HEOR workflows.
- Add ecosystem integrations through optional extras only after stable public
  APIs and fixture contracts exist.
- Require smoke CI, Renovate coverage, security checks, documentation, and a
  removal path for each optional adapter dependency.
- Do not require changes to the `mars` core API. Add adapter logic in
  `innovate` or a future companion package if needed.

## Promotion Criteria

1. Document artifact contracts and dependency policy.
2. Add deterministic compatibility fixtures.
3. Add experimental adapters behind optional extras.
4. Add cross-repo smoke CI and fixture validation.
5. Promote only after version compatibility, docs, release notes, and
   deprecation policy are clear.

## Immediate Follow-Up

- Define a minimal adoption-trajectory fixture that `lifecourse` can consume.
- Define a diffusion-uncertainty fixture that `voiage` can use for VOI examples.
- Decide whether HEOML extension schemas should live in `innovate`, `lifecourse`
  while HEOML is embedded, or a future standalone `heoml` repository.
- Benchmark whether `mars` improves adoption-curve surrogate workflows before
  exposing it as a supported optional backend.
