# HEOML Innovate Extension Schemas

This directory is the interim schema home for `heoml.extensions.innovate`:

`specs/ecosystem/heoml/extensions/innovate/`

The schemas in this namespace describe how `innovate` adoption, uptake,
diffusion, policy-spread, diagnostics, and provenance artifacts can be wrapped
for HEOR bundles while HEOML remains an ecosystem direction rather than a
standalone `heoml` repository.

## Interim Placement

The repo-local home is intentionally temporary. It keeps schema drafts near the
`innovate` functional kernel, artifact fixtures, and Arrow-compatible
interchange checks while avoiding a dependency on `lifecourse` or other sibling
projects.

The migration trigger is a standalone `heoml` repository that provides a
published semver schema bundle, stable `heoml.extensions.innovate` namespace,
cross-repository fixture CI, and a documented deprecation window for repo-local
schema definitions.

## Contract Rules

- Every schema must expose a `schema_version`.
- Versioning must follow semver-compatible evolution within a major version.
- Manifests and small metadata must use binding-friendly JSON.
- Tabular payloads must use Arrow-compatible columns, scalar types, and
  nullability.
- JSON Schema should validate manifests and metadata when artifacts cross
  repository boundaries.
- HEOML wrappers may add namespace and health-economic bundle metadata, but the
  underlying `innovate` tabular payload must remain identical.
- Contracts MUST NOT use private Python objects.
- Contracts MUST NOT use pickle.
- Contracts MUST NOT use private Python object framing.

## Initial Artifact Families

- `adoption_curve`
- `uptake_trajectory`
- `policy_spread_trace`
- `network_diffusion_trace`
- `diagnostics_record`
- `provenance_record`

Concrete schemas will be added only with deterministic fixtures and compatibility
checks. Until then, this directory records placement, ownership, versioning, and
migration governance.

## Paired Bass outcomes for downstream decisions

The documented `diffusion_v1` fixture supplies strategy, draw, time and
parameter identities. A paired export uses `scenario_id` as the strategy
key and retains the same ordered draw IDs and time grid for each strategy.
Each trajectory row must resolve to the parameters for its own
`(scenario_id, draw_id)`. The fixture varies parameter values by strategy,
so pairing requires matching draw identity and parameter names and units.
Parameter values remain strategy-specific.

The positive oracle uses the fixture's two strategies, draws 1 through 3,
and times 0 through 3. Negative cases remove a draw, swap two draw blocks,
duplicate a parameter key, or change one strategy's declared time unit
or economic perspective. Validate those conditions before returning the
artifact. Preserve the original numeric rows and order on success.

A runtime producer must declare its time unit and provenance. Economic
perspective, when supplied, belongs to downstream HEOR metadata and must
agree across compared strategies. The historical CSV fixture alone
cannot verify these declarations. Innovate supplies adoption outcomes,
the HEOR producer supplies economic values, and Voiage performs VOI.

Call `validate_paired_outcomes` with a schema 2.0 kernel request. Its
payload contains `strategies`, each with `scenario_id`, `time_unit`,
`provenance` as a nonblank string, optional `perspective`, and ordered
`draws`. Each draw has an unsigned `draw_id`, aligned `time` and
`adoption` arrays, and `parameters` entries with `name`, `unit` and
numeric `value`. The validator returns the supplied payload on success.
