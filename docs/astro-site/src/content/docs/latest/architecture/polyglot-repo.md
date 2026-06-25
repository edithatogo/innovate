---
title: Polyglot Repository Architecture
description: Documentation and ownership map for the current repository layout.
slug: latest/architecture/polyglot-repo
---

# Polyglot Repository and Documentation Architecture

This page is the navigation and ownership map for the polyglot `innovate`
repository. It keeps the current source layout stable while making the docs
clear for users, binding authors, HPC administrators, and maintainers.

No source tree move is required for the current release. The existing paths
remain canonical until a future track provides a migration plan, redirects, and
backward-compatible link checks.

## Audience Navigation

**Users** should start with kernel, tutorials, and bindings pages. These pages
explain the stable functional kernel, primary examples, and language-specific
entry points without requiring repository-layout knowledge.

**Binding authors** should use `docs/source/bindings.rst`,
`docs/astro-site/src/content/docs/maintainers/publication.md`, and the binding
README files under `bindings/*/README.md`. Binding work must stay thin over the
kernel contract and preserve package-manager metadata for each language.

**HPC administrators** should use
`docs/astro-site/src/content/docs/operations/scientific-hpc.md`,
`docs/astro-site/src/content/docs/operations/xla-backend.md`, and
`docs/astro-site/src/content/docs/operations/rust-core.md`. These pages
distinguish deployment evidence, accelerator evidence, and native-core
promotion from public binding APIs.

**Maintainers** should use ADRs, release notes policy, publication gates, and
the Conductor track archive. Maintainer material should record why a path
exists, who owns it, and which release gate protects it.

## Target documentation architecture

The target architecture is a docs-only reorganization around five stable entry
points. It adds clearer navigation without renaming source directories or
package surfaces.

**Core contract** pages cover kernel, schema, stability, Arrow interchange,
diagnostics, and API reference behavior shared by every language.

**Binding packages** documentation links to R, Rust, Julia, TypeScript, Go, and
C# binding package docs, then delegates package-manager details to
`docs/astro-site/src/content/docs/maintainers/publication.md`.

**HPC deployment** material remains in readiness and strategy pages until Spack,
EasyBuild, scheduler, or accelerator artifacts are implemented by their own
tracks.

**Submission evidence** should link to stable docs, CI artifacts, release notes,
and archived Conductor evidence instead of duplicating those sources.

**Maintainer decisions** through ADRs, release policy, publication gates, and
Conductor archive entries record ownership decisions and migration rationale.

## Ownership Map

| Area | Canonical paths | Ownership rule |
| --- | --- | --- |
| Core package | `src/innovate/`, `docs/astro-site/src/content/docs/architecture/polyglot-repo.md`, `docs/source/innovate.kernel.rst`, `docs/source/api_reference.rst` | Python remains the reference user API while the functional kernel and schema contract define portable behavior. |
| Language bindings | `bindings/r/README.md`, `bindings/rust/README.md`, `bindings/julia/README.md`, `bindings/typescript/README.md`, `bindings/go/README.md`, `bindings/csharp/README.md`, `docs/source/bindings.rst` | R, Rust, Julia, TypeScript, Go, and C# bindings are language-owned adapters over the same kernel contract, not independent model forks. |
| Packaging and release | `docs/astro-site/src/content/docs/maintainers/publication.md`, release policy docs, package manifests, and release workflows | Package-manager metadata and release gates are owned with each binding, then checked by the shared publication documentation and tests. |
| Scientific and HPC ecosystem | `docs/astro-site/src/content/docs/operations/scientific-hpc.md`, `docs/astro-site/src/content/docs/operations/xla-backend.md`, `docs/astro-site/src/content/docs/operations/rust-core.md` | HPC, accelerator, ABI, and Rust-core evidence lives in docs until a dedicated implementation track promotes code or packaging artifacts. |
| Community submission dossiers | Future dossier docs and Conductor archives that reference reviewer evidence | Submission material should cite stable docs and release artifacts rather than duplicate package metadata. |

## Source-Layout Guidance

### Repository layout decision

Source tree moves are deferred. The current release needs documentation
navigation and ownership clarity, not directory churn. Any later source move
must be justified by a dedicated track with tests and redirect coverage.

Keep the current source layout. The repository already separates concerns:

* `src/innovate/` owns the Python public API, reference implementation, and
  stable functional kernel surface.
* `bindings/` owns language package surfaces and package-manager metadata.
* `docs/astro-site/` owns the canonical Starlight site, user docs, release docs,
  roadmaps, and reviewer-facing architecture decisions.
* `docs/source/` remains a legacy compatibility and API-generation area until
  the final Sphinx retirement slice removes or regenerates every RST page.
* `conductor/tracks/` owns active plans. `conductor/archive/` owns completed
  evidence for decisions and release gates.

Future source moves need a separate track before implementation. That track must
identify the old path, new path, public docs link, release artifact impact, and
package-manager impact before any files move.

## Migration And Redirect Rules

Existing paths remain canonical unless the migration track proves a move is
necessary. A move is acceptable only when all of these conditions are met:

* the old docs page keeps a stable redirect or explicit forwarding stub;
* the new page is added to the Starlight sidebar before the old path is removed;
* tests cover the old path, new path, and affected audience navigation;
* package files, generated artifacts, and publication gates are not renamed as
  part of a docs-only cleanup;
* release notes call out any user-visible path change.

These rules protect existing links from README files, package registries,
release artifacts, Conductor archives, and external reviewer dossiers.

Canonical source:

* `docs/astro-site/src/content/docs/architecture/polyglot-repo.md`
