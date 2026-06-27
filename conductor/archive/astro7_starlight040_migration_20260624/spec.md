# Astro 7 and Starlight 0.40 Dedicated Migration

## Overview

Create a dedicated implementation track for the Astro 7 and Starlight 0.40 documentation migration because it is the largest frontend/docs bleeding-edge dependency jump in the modernization roadmap. This track separates the docs-site migration risk from general dependency modernization, Python 3.14 work, release hardening, and Rust-core work.

The track must treat the documentation site as a product surface with its own dependency contract, plugin compatibility evidence, route validation, CI gates, and release-readiness artifacts. The user-requested target is "Astro 7/Starlight 0.40"; if the repository already contains a later Starlight line such as 0.41.x, implementation must either return to the selected 0.40.x target or explicitly promote the later line with evidence and update the documented baseline.

## Functional Requirements

- Establish a dedicated Astro/Starlight migration contract for `docs/astro-site`.
- Audit and record the exact current versions of `astro`, `@astrojs/starlight`, `@astrojs/markdown-remark`, `starlight-links-validator`, `starlight-versions`, `@astrojs/starlight-docsearch`, `starlight-polyglot`, `@astrojs/check`, TypeScript, pnpm, and any transitive peer-dependency exceptions.
- Reconcile the requested Starlight 0.40 target with the currently committed Starlight manifest.
- Update the docs tech-stack, migration documentation, and release-readiness evidence so the Astro 7/Starlight jump is visibly tracked as a dedicated frontend/docs migration.
- Ensure the Starlight site validates the active route inventory, versioned `latest/` content, sidebars, redirects, `/404`, generated Python API docs, custom CSS, DocSearch gating, and link validation.
- Ensure Sphinx remains legacy/archive-only unless explicitly required as a redirect-reference source.
- Capture external or ecosystem compatibility constraints instead of silently lowering or weakening the Astro/Starlight target.
- Add or update tests that fail if the documented Astro/Starlight baseline, plugin compatibility matrix, migration manifest, or docs evidence drift from `docs/astro-site/package.json` and `pnpm-lock.yaml`.

## Non-Functional Requirements

- Preserve docs user experience and route stability while making the bleeding-edge dependency jump explicit.
- Keep the migration reproducible through pnpm and CI.
- Prefer explicit version evidence over broad semver claims for the frontend/docs stack.
- Avoid changing Python package runtime dependencies as part of this track unless they are required for Starlight polyglot generation.
- Avoid publishing or claiming external service readiness for DocSearch unless required credentials and receipts exist.

## Acceptance Criteria

- `docs/astro-site/package.json`, `docs/astro-site/pnpm-lock.yaml`, and the documented tech-stack agree on the selected Astro 7/Starlight target line.
- A migration manifest or equivalent evidence artifact records the selected target, plugin versions, peer-dependency exceptions, route inventory, and validation commands.
- Unit tests validate the Astro/Starlight dependency contract and fail on version or plugin drift.
- `pnpm --dir docs/astro-site check` passes.
- `pnpm --dir docs/astro-site build` passes.
- `uv run nox -s docs` passes.
- Release-readiness evidence for docs build and compatibility is refreshed after the migration.
- The Conductor plan records a commit after every task, automated review after every phase, push and GitHub Actions monitoring after every phase and track, and a final track review before archival.

## Out Of Scope

- Rewriting the Python library API.
- Rust-core migration work.
- External registry publication.
- Replacing the entire visual design system unless required by Astro/Starlight compatibility.
- Claiming DocSearch production readiness without external credentials and deployment evidence.
