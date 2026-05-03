# Specification: HEOML Schema Placement Decision

## Overview

Decide where HEOML extension schemas should live while `heoml` is still an
ecosystem direction rather than a standalone stable repository. This track
converts the ecosystem incubation follow-up "Decide whether HEOML extension
schemas should live in `innovate`, `lifecourse` while HEOML is embedded, or a
future standalone `heoml` repository" into a narrow governance track.

## Roadmap Source

- `docs/ecosystem/module_incubation_strategy.md`
- `specs/ecosystem/README.md`
- Ecosystem incubation follow-up: HEOML extension schema placement

## Functional Requirements

1. Compare three placement options: repo-local `innovate` schemas, embedded
   `lifecourse` schemas, and a future standalone `heoml` repository.
2. Define ownership, versioning, compatibility, publication, and migration
   consequences for each option.
3. Recommend an interim placement that keeps `innovate` artifact schemas
   portable and avoids private sibling-project internals.
4. Define migration rules if the schemas later move to a standalone HEOML
   repository.
5. Add documentation or ADR-style decision material that downstream projects
   can reference.

## Non-Functional Requirements

1. The decision must preserve `innovate` as the adoption/diffusion artifact
   producer, not the owner of all HEOR simulation semantics.
2. Schema placement must support semver-compatible evolution and deprecation
   windows.
3. The decision must not require sibling-project dependencies in the base
   install.
4. The decision must preserve binding-friendly JSON and Arrow-compatible
   contract surfaces.

## Acceptance Criteria

1. A documented placement decision exists with rationale and alternatives.
2. The ecosystem specs name the selected interim schema home and migration
   trigger.
3. Tests or documentation checks guard that HEOML extension schemas are not
   described as private Python object contracts.
4. Release and versioning expectations are explicit enough for future
   cross-repository fixtures.

## Out of Scope

1. Creating a new external `heoml` repository.
2. Implementing all HEOML extension schemas.
3. Moving existing sibling-project code.
4. Publishing cross-repository packages.
