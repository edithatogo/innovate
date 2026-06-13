# Starlight + starlight-polyglot Migration Specification

## Overview

Replace the legacy Sphinx documentation pipeline at `docs/source/` with an
Astro Starlight site at `docs/astro-site/` that uses `starlight-polyglot` to
auto-generate Python API documentation from the `src/innovate/` source tree via
Griffe docstring extraction.

## Scope

- **In scope:**
  - Astro/Starlight scaffold at `docs/astro-site/` (already partially present
    with 41 MDX pages).
  - Configuration and integration of `starlight-polyglot` as a Starlight plugin
    configured for Python with `entryPoints: ['src/innovate']`.
  - Upgrade of `@astrojs/starlight` from `0.38.4` to `^0.39.0` to satisfy
    `starlight-polyglot` peer dependency.
  - CI/CD workflow replacement: `.github/workflows/docs.yml` now builds the
    Starlight site via pnpm instead of Sphinx via uv.
  - Conductor tech-stack update: Sphinx entries replaced with Starlight +
    starlight-polyglot.
  - Track documentation under
    `conductor/tracks/starlight_migration_20260513/`.

- **Out of scope:**
  - Migration of individual RST pages to MDX (41 pages already exist as MDX).
  - Content reorganisation or rewriting of existing handbook pages.
  - Sphinx source removal: `docs/source/` and `docs/build/` are retained as
    archival references.

## Key Decisions

1. **Starlight plugin, not Astro integration** — `starlight-polyglot` exposes a
   `StarlightPlugin`, so it is injected via the Starlight `plugins` array in
   `astro.config.mjs`, not as a top-level Astro integration.
2. **Entry point path** — `src/innovate` is passed as the Python entry point
   relative to the repository root; the polyglot handler invokes griffe
   internally at build time.
3. **Output directory** — Generated API MDX pages are written to
   `api/python/` under the Starlight content root.
4. **Workspace dependency** — `starlight-polyglot` is referenced via a local
   file path (`file:/Users/doughnut/GitHub/starlight-polyglot/packages/starlight-polyglot`)
   in `package.json`.

## Verification Criteria

- [ ] `pnpm install` completes without errors in `docs/astro-site/`.
- [ ] `pnpm build` generates the site including API docs from `src/innovate/`.
- [ ] All 41 existing MDX handbook pages render correctly.
- [ ] CI workflow triggers on push to `work`/`main` and builds via pnpm.
- [ ] `workflow_dispatch` deployment publishes to GitHub Pages.
