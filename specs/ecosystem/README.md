# Ecosystem Contracts

This directory is reserved for health economics and outcomes research (HEOR)
ecosystem-level contracts that connect `innovate` to sibling projects without
importing their internals.

Initial contract targets:

- `lifecourse`: health-intervention adoption and implementation uptake trajectories for
  health-economic scenarios.
- `voiage`: health-intervention adoption and diffusion uncertainty artifacts
  for HEOR VOI workflows.
- `mars`: surrogate/metamodel artifact references and backend metadata for
  optional response-surface workflows.
- HEOML: health-economic extension namespace for uptake, adoption, diffusion,
  and policy-spread artifacts.

Non-goals:

- importing sibling-project internals
- using pickle as a portable contract
- changing the mars core API
- turning `innovate` into a health-economic simulation or VOI engine

Stable contracts should define:

- producer and consumer responsibilities
- schema versions
- required and optional artifacts
- compatibility fixtures
- dependency and optional-extra policy
- diagnostics and provenance fields
- deprecation and migration rules
- explicit promotion stages from documented to experimental to supported
- the smoke CI, Renovate, security, and documentation gates needed for each
  optional adapter

Expected artifact groups include:

- adoption curves and uptake trajectories
- policy-spread traces and network diffusion traces
- decision-analysis and operational-model bundles
- discrete-event event traces, queue metrics, and pathway logs
- diagnostics, calibration outputs, and provenance records
- Arrow or Parquet tabular payloads plus JSON manifests and schemas

Operational modeling fixtures:

- [operational_modeling/treeage_style/manifest.json](./operational_modeling/treeage_style/manifest.json)
  defines a TreeAge-style decision-tree and state-transition artifact shape for
  HTA and reimbursement examples. It records strategy, state, transition,
  payoff, schema-version, provenance, and XLA eligibility or rejection metadata
  without requiring proprietary TreeAge parsing.
- [operational_modeling/des/manifest.json](./operational_modeling/des/manifest.json)
  defines a DES event-log and queue-metric artifact shape for patient pathway
  examples. It records run metadata, deterministic ordering rules, resources,
  pathway states, event rows, queue metrics, provenance, and XLA eligibility or
  rejection metadata without requiring private engine state.

These fixture contracts sit at the documented step of the adapter promotion ladder.
Runtime simulation engines out of the current `innovate` package remain out of
scope until an adapter has explicit optional extras, smoke CI, security coverage,
compatibility matrices, documentation, and a removal path.

HEOML alignment should treat `innovate` artifacts as the portable base layer
and only add extension metadata when a health-economic bundle requires a shared
namespace or cross-repo wrapper.

HEOML schema placement:

- interim schema home: `specs/ecosystem/heoml/extensions/innovate/`
- selected namespace: `heoml.extensions.innovate`
- decision record: `docs/adr/0005-heoml-schema-placement.md`
- migration trigger: a standalone `heoml` repository with a published semver
  schema bundle, fixture CI, a stable extension namespace, and a documented
  deprecation window for repo-local schemas
- contract surface: `schema_version` fields, binding-friendly JSON manifests,
  JSON Schema validation, and Arrow-compatible tabular payloads
- exclusions: schemas MUST NOT use private Python objects, MUST NOT use pickle,
  and MUST NOT use private Python object framing

Namespace guidance:

- keep `innovate` artifact names for repo-local consumers
- use `heoml.extensions.innovate` for cross-repo HEOR bundles
- preserve the same underlying tabular payload across both representations

Dependency policy:

- keep the base install free of sibling-project dependencies
- expose adapters only through optional extras or equivalent explicit flags
- require deterministic fixtures and smoke CI before promotion
- require a compatibility matrix before an adapter is marked supported

Operational modelling notes:

- TreeAge-style decision-tree and state-transition adapters belong in the
  ecosystem contract when they consume or emit HEOR adoption and pathway
  artifacts.
- DES adapters should represent pathways as event logs or simulation run
  bundles, not private engine state.

Current scaffolds:

- [lifecourse/adoption_trajectory/v1/manifest.json](./lifecourse/adoption_trajectory/v1/manifest.json)
  for the documented adoption-trajectory fixture that `lifecourse` can inspect
  through Arrow or Parquet without importing `innovate` internals
- [voiage/uncertainty/diffusion_v1/manifest.json](./voiage/uncertainty/diffusion_v1/manifest.json)
  for the documented diffusion-uncertainty fixture that `voiage` can consume
  in HEOR VOI examples without importing sibling-project internals
- [process/README.md](./process/README.md) for the HEOR process-mining outline
  and PM4Py ecosystem-only contract
- [Conductor track: HEOR Module Naming Brainstorm](../../conductor/archive/heor_module_naming_brainstorm_20260429/)
  for the reserved sibling-module naming shortlist and interface expectations
