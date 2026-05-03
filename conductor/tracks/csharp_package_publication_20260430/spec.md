# Specification: C# Package Publication

## Overview

Prepare C# package publication through NuGet only after the thin-binding contract, .NET 10/.NET 11 setup, schema compatibility, and CI validation are stable. This track turns the roadmap item "C# package publication before the thin-binding contract is validated" into a concrete publication-readiness track.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- Deferred work item: "C# package publication before the thin-binding contract is validated"

## Functional Requirements

1. Validate the C# binding contract against the functional kernel schemas and cross-language fixtures.
2. Ensure C# projects target the repository's required .NET 10 and .NET 11 configuration.
3. Add NuGet package metadata, versioning, license, README, and source-link expectations.
4. Add CI jobs for restore, build, test, pack, and publication dry runs.
5. Define release gates for signing, provenance, artifact retention, and package manager publishing.
6. Document publication steps and rollback expectations.

## Non-Functional Requirements

1. C# must remain a thin binding and must not duplicate core model logic.
2. Publication automation must avoid pushing to NuGet on normal pull-request validation.
3. Versioning must align with the broader ecosystem release strategy.
4. Package artifacts must include enough metadata for downstream consumers.

## Acceptance Criteria

1. C# package metadata and .NET 10/.NET 11 build configuration are publication-ready.
2. CI validates C# restore, build, test, pack, and dry-run publication behavior.
3. NuGet publishing is documented and gated behind explicit release conditions.
4. Schema compatibility tests prove the C# package still targets the thin-binding contract.

## Out of Scope

1. Publishing to NuGet before package validation passes.
2. Implementing independent C# model algorithms.
3. Changing the canonical kernel contract solely for C# convenience.
