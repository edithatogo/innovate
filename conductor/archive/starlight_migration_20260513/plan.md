# Migration Plan

## Status: Complete

This track documents the completed migration from Sphinx to Starlight +
starlight-polyglot for the innovate documentation site.

## Tasks Completed

| # | Task | File(s) | Status |
|---|------|---------|--------|
| 1 | Upgrade `@astrojs/starlight` to `^0.39.0` | `docs/astro-site/package.json` | Done |
| 2 | Add `starlight-polyglot` workspace dependency | `docs/astro-site/package.json` | Done |
| 3 | Import and configure `starlight-polyglot` with Python entry points | `docs/astro-site/astro.config.mjs` | Done |
| 4 | Replace Sphinx build in CI/CD workflow with pnpm/Starlight | `.github/workflows/docs.yml` | Done |
| 5 | Update tech-stack.md: Sphinx → Starlight + starlight-polyglot | `conductor/tech-stack.md` | Done |
| 6 | Create migration track metadata, spec, and plan | `conductor/tracks/starlight_migration_20260513/` | Done |

## Verification Steps

1. Run `pnpm install` in `docs/astro-site/` to install all dependencies.
2. Run `pnpm build` to verify the site builds with polyglot-generated API docs.
3. Confirm that the 41 existing MDX pages at `src/content/docs/` render
   alongside the new API pages under `api/python/`.
4. Trigger the `Deploy Documentation` workflow via `workflow_dispatch` to
   publish the Starlight site to GitHub Pages.

## Rollback

If issues are encountered:
- Revert `docs/astro-site/package.json` to restore `@astrojs/starlight` at
  `0.38.4` and remove the `starlight-polyglot` entry.
- Revert `docs/astro-site/astro.config.mjs` to the original `starlight(starlightConfig)` call.
- Revert `.github/workflows/docs.yml` to the Sphinx-based build.
- Revert `conductor/tech-stack.md` to the original Sphinx/Starlight entries.
- Archive this track by moving `conductor/tracks/starlight_migration_20260513/`
  to `conductor/archive/`.
