# Astro/Starlight documentation site

This directory contains the active Astro/Starlight documentation site surface
for the `innovate` repository.

Baseline decisions:

- `astro` `^7.0.5`
- `@astrojs/starlight` `^0.41.3`
- `@astrojs/markdown-remark` `^7.2.0`
- `starlight-versions` `0.9.1`
- `starlight-links-validator` `0.25.2`
- `@astrojs/starlight-docsearch` `0.7.0`
- `@astrojs/sitemap` `^3.7.2`

`starlight-versions` is enabled with the existing `latest/` versioned content,
and the active build validates Astro 7 non-doc routes such as `/404`.

Cutover policy:

- Astro/Starlight is the only documentation stack
- Algolia DocSearch as the selected search provider
- route preservation through Astro route evidence and cutover verification
- `pnpm build && pnpm check` as the active Starlight build gate
- `python ../../scripts/verify_production_docs.py --json` after `pnpm build`
  to verify production routes, sitemap, search fallback, versioned
  docs, generated API pages, and CI wiring

Key artifacts:

- `astro.config.mjs`
- `starlight.config.mjs`
- `pnpm-lock.yaml`
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
