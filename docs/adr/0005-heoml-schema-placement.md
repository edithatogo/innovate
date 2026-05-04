# ADR 0005: HEOML Schema Placement

- Status: Accepted
- Date: 2026-05-04

## Context

`innovate` needs a durable location for HEOML extension schemas while HEOML is
still an ecosystem direction, not a standalone stable repository. These schemas
describe how adoption, uptake, diffusion, policy-spread, diagnostics, and
provenance artifacts can be wrapped for health-economic bundles without making
`innovate` own health-economic simulation semantics.

The placement has to support future Python, R, Rust, Julia, TypeScript, Go, and
C# bindings. It therefore has to stay based on binding-friendly JSON,
JSON Schema, and Arrow-compatible tabular payloads. It MUST NOT use private
Python objects, pickle, or private Python object framing as a contract.

The decision compares ownership, versioning, compatibility, publication, and
migration consequences for each placement option.

## Decision

The interim schema home is:

`specs/ecosystem/heoml/extensions/innovate/`

This keeps the HEOML extension profile repo-local to `innovate` while the
schemas are being designed, validated against fixtures, and aligned with the
functional kernel and Arrow-compatible artifact contracts.

The migration trigger is the creation of a standalone `heoml` repository with:

1. A published semver package or schema bundle.
2. A stable `heoml.extensions.innovate` namespace.
3. CI that validates the same JSON Schema and Arrow-compatible fixtures against
   `innovate`.
4. A documented deprecation window for repo-local schemas.
5. Cross-repository ownership and release responsibilities.

After that trigger, `innovate` will keep compatibility aliases or pointers for
at least one minor-release deprecation window before removing repo-local schema
definitions.

## Options Considered

| Option | Ownership | Versioning | Compatibility | Publication | Migration |
| --- | --- | --- | --- | --- | --- |
| repo-local `innovate` schemas | `innovate` owns adoption/diffusion extension schemas while they mature. | Use `schema_version` fields and semver-compatible changes in repo releases. | Directly test against kernel payloads, JSON Schema, and Arrow-compatible fixtures. | Ship with the repo docs/specs; no base dependency added. | Move to standalone HEOML after the migration trigger and deprecation window. |
| embedded `lifecourse` schemas | `lifecourse` owns schema text while HEOML is embedded in its workflows. | Tied to `lifecourse` releases, which may not align with `innovate` artifacts. | Risks coupling adoption artifacts to simulation internals. | Published only through `lifecourse` documentation or package assets. | Requires extracting adoption schemas back out before other consumers can rely on them. |
| future standalone `heoml` repository | HEOML owns shared extension semantics across ecosystem projects. | Clean semver package or schema-bundle releases. | Best long-term cross-project contract if fixture CI exists. | Published independently for all sibling projects and language consumers. | Target end-state once repository, package, CI, and governance exist. |

## Consequences

### Positive

- `innovate` can validate HEOML extension schemas against its own public
  artifacts without depending on sibling-project internals.
- Cross-language bindings can consume binding-friendly JSON manifests and
  Arrow-compatible tabular payloads rather than Python objects.
- The future standalone `heoml` repository has a clear migration trigger,
  deprecation window, and compatibility baseline.

### Negative

- `innovate` temporarily hosts schemas that are not part of its core modeling
  API.
- A later standalone HEOML extraction will require compatibility aliases,
  release notes, and cross-repository CI.

## Governance Rules

1. HEOML extension schemas in this repo MUST include a `schema_version` field.
2. Schema evolution must be semver-compatible within a major version.
3. Tabular artifacts MUST use Arrow-compatible columns and scalar types.
4. Metadata and manifests MUST use binding-friendly JSON and JSON Schema.
5. Contracts MUST NOT use private Python objects, pickle, or private Python
   object framing.
6. HEOML wrappers may add namespace and health-economic bundle metadata, but
   they must not alter the underlying `innovate` tabular payload.

## Follow-Up Work

1. Add minimal schema fixtures for the first `heoml.extensions.innovate`
   adoption and uptake artifacts.
2. Add cross-repository fixture checks once a standalone `heoml` repository
   exists.
3. Publish migration notes when schema ownership moves out of this repo.
