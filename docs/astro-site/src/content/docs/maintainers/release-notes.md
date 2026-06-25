---
title: Release Notes
description: Release-notes policy and changelog responsibilities.
---

# Release Notes

This repository uses GitHub release automation for published release notes and
keeps `CHANGELOG.md` as the durable in-repository summary of major user-facing
changes.

## Responsibilities

**Release Please** owns GitHub release creation from conventional commits on
`main`. It is the canonical automation path for release titles, tags, and GitHub
release notes.

**Release Drafter** maintains a draft view of unreleased changes for
maintainers. It is a preview and triage aid, not the source of truth for
published release notes.

**Commitizen** enforces and assists conventional commit workflows locally. Its
`CHANGELOG.md` integration is useful for local release preparation, but CI
release publication is handled by Release Please.

**`CHANGELOG.md`** records concise release summaries that are useful outside
GitHub, especially for package-manager users. It should include the current
aligned package version and any major package, API, binding, CI, or publication
changes.

## Package Manager Expectations

Every package publication target should have release notes that identify the
aligned version and any language-specific caveats:

- PyPI and TestPyPI use the primary Python package metadata and GitHub release.
- npm publishes `innovate.ts` with the aligned version and npm package metadata.
- crates.io publishes `innovate-rs` with crate metadata and Rust MSRV notes.
- R-universe and CRAN candidates use the R source package, PDF manual artifact,
  and vignette validation notes.
- Julia General submissions use the Julia project version and registry
  compatibility notes.
- Go modules use the `bindings/go/vX.Y.Z` submodule tag convention.
- NuGet publishes `innovate.cs` with package metadata, readme, symbols, and
  target-framework notes for .NET 10 and .NET 11.

## Version Synchronization

The canonical release version comes from `pyproject.toml`. The repository keeps
package manifests aligned with that source through `scripts/sync_versions.py`:

- `python scripts/sync_versions.py --check` fails CI when a supported package
  manifest drifts from the canonical release version.
- `python scripts/sync_versions.py --write` rewrites the supported package
  manifests to match the canonical version during a version bump.
- The script covers the Python, Julia, Rust, R, and C# package manifests so
  maintainers do not have to hand-edit the same version in multiple places.
- `nox -s version_sync` runs the same guard locally, and
  `nox -s version_sync -- --write` rewrites the supported manifests during
  release prep.

## Drift Guards

Static tests assert that the current package version appears in `CHANGELOG.md`,
that this policy is linked from the Starlight docs, that the version sync guard
is documented, and that release automation comments point back to this policy
instead of claiming the policy is undefined.
