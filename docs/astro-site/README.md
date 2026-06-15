# Astro/Starlight documentation site

This directory contains the active Astro/Starlight documentation site surface
for the `innovate` repository. Legacy Sphinx source is retained outside this
directory as archival and redirect-reference material.

Baseline decisions:

- `@astrojs/starlight` `0.38.4`
- `starlight-versions` `0.9.0`
- `starlight-links-validator` `0.24.0`
- `@astrojs/starlight-docsearch` `0.7.0`
- `astro` `^6.0.0`
- `@astrojs/sitemap` `^3.7.2`

`starlight-versions` is enabled with the existing `latest/` versioned content,
and the active build validates Astro 6 non-doc routes such as `/404`.

Cutover policy:

- Astro/Starlight is the active documentation stack
- legacy Sphinx URLs are preserved as compatibility aliases
- Algolia DocSearch as the selected search provider
- route preservation through redirect evidence and cutover verification
- `pnpm build && pnpm check` as the active Starlight build gate
- `python ../../scripts/verify_production_docs.py --json` after `pnpm build`
  to verify production routes, redirects, sitemap, search fallback, versioned
  docs, generated API pages, and CI wiring

Key artifacts:

- `astro.config.mjs`
- `starlight.config.mjs`
- `package-lock.json`
- `src/content.config.ts`
- `src/content/docs/`
- `../source/_static/astro_starlight/migration_manifest.json`
- `../source/_static/astro_starlight/content_inventory.json`
- `../source/_static/astro_starlight/redirect_inventory.json`
- `../source/_static/astro_starlight/route_coverage.json`
- `../source/_static/astro_starlight/cutover_verification.json`
- `../source/_static/astro_starlight/link_validation_report.json`
- `../source/_static/astro_starlight/production_docs_verification.json`
- `../source/_static/astro_starlight/generate_route_coverage.py`
- `../source/_static/astro_starlight/generate_cutover_verification.py`
- `../source/_static/astro_starlight/generate_link_validation.py`
