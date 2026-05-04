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
