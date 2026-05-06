# Specification

## Overview

Make the repository versioning system more automated and less drift-prone by
anchoring the release version in one canonical source and verifying that all
language package manifests, release docs, and CI gates stay aligned with it.

The existing release flow already uses SemVer, release-please, commitizen, and
language-specific package metadata. This track hardens that setup by adding a
single synchronization/check path so version drift is detected before release.

## Functional Requirements

1. Provide a canonical release-version source that the repository can read for
   checks and synchronization.
2. Add a version sync/check script that validates the current release version
   against the package manifests and can rewrite them when the manifest changes.
3. Cover the primary release-bearing files:
   - `pyproject.toml`
   - `bindings/rust/Cargo.toml`
   - `bindings/julia/Project.toml`
   - `bindings/typescript/package.json`
   - `bindings/r/DESCRIPTION`
   - `bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj`
4. Keep existing release automation behavior intact while adding a stronger
   drift guard.
5. Update repository documentation so maintainers know how the version source
   and sync check work.

## Non-Functional Requirements

1. The sync check must be deterministic and safe to run in CI.
2. The script must support a non-mutating verification mode.
3. The versioning mechanism should avoid adding new manual version duplication
   where the existing release flow already has a source of truth.
4. The change should preserve compatibility with the current release-please and
   commitizen workflow.

## Acceptance Criteria

1. A canonical version sync/check tool exists in the repository.
2. CI fails if the supported package manifests drift from the canonical version.
3. The repository docs explain the version source and how to refresh the
   manifests.
4. The tracked release-bearing package manifests stay aligned with the current
   release version.
5. The track can be archived cleanly once the sync guard and docs are in place.

## Out of Scope

1. Replacing release-please or commitizen.
2. Building a custom multi-package release orchestrator.
3. Changing the versioning semantics of the individual language ecosystems.
