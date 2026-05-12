# Specification: Astro/Starlight Documentation Site Migration

## Overview

Migrate the documentation site from the current Sphinx-centric delivery model
to a future Astro/Starlight site while preserving the current docs content,
canonical navigation, and link stability. The track should make the Starlight
version baseline explicit, decide the cutover policy up front, and choose a
small plugin set that keeps versioned docs, link validation, search, sitemap
generation, and redirects auditable.

## Background

The repository already treats the Sphinx site as canonical for the present
release, and the Starlight roadmap track has been archived after documenting
the current version baseline. This track is the implementation track that uses
that baseline to build the actual Astro/Starlight docs surface and migration
plan. The migration must record whether the cutover is a parallel run or a
full replacement so the acceptance boundary is unambiguous.

## Recommended Baseline

The initial Starlight baseline should remain explicit and version-pinned:

- `@astrojs/starlight` `0.38.4`
- `starlight-versions` `0.5.4`
- `starlight-links-validator` `0.24.0`
- `@astrojs/starlight-docsearch` `0.6.1` if Algolia DocSearch is selected

Recommended additional Astro integration:

- `@astrojs/sitemap` to generate a crawlable sitemap for the public docs site

## Functional Requirements

1. Scaffold an Astro/Starlight docs site that can build from the current
   documentation content.
2. Decide and document the migration mode: parallel-run or full cutover.
3. Maintain a content inventory that maps each Sphinx page to a future Astro
   route, redirect, or archive-only status.
4. Version the docs navigation with `starlight-versions`.
5. Validate internal links with `starlight-links-validator`.
6. Integrate search using `@astrojs/starlight-docsearch` if Algolia is chosen,
   or record the explicit alternative if another provider is selected.
7. Generate a sitemap with `@astrojs/sitemap` or an equivalent official Astro
   integration.
8. Preserve canonical URLs or redirects for existing Sphinx docs paths and
   keep a redirect inventory synchronized with the content inventory.
9. Update docs and tests so the new site, versions, plugins, redirects, and
   route stability checks are explicit and auditable.

## Non-Functional Requirements

1. The migration must not break the current documentation availability during
   the transition.
2. Version pins must be easy to update when the ecosystem changes.
3. The track must not over-specify unnecessary plugins.
4. The final site must remain maintainable and link-checkable in CI.
5. The migration mode and search-provider choice must be explicit before
   content migration begins.
6. The redirect inventory and content inventory must remain synchronized.

## Acceptance Criteria

1. The Astro/Starlight site builds successfully with the pinned baseline.
2. Versioned navigation, link validation, and search integration are present
   or explicitly deferred with rationale.
3. Sitemap generation is configured or explicitly deferred with rationale.
4. Redirects or stable forwarders protect the existing Sphinx doc URLs.
5. The migration docs and tests reflect the same version/plugin baseline.
6. The content inventory, redirect inventory, and route stability checks all
   agree on the final migration shape.
7. The track can be archived cleanly when the docs-site migration is complete.

## Out of Scope

1. Rust-core runtime changes.
2. New product features unrelated to documentation.
3. Rewriting docs content for style only, unless required by the migration.
4. Changes to the current code runtime or package APIs.
