# Astro/Starlight migration scaffold

This directory contains the planned Astro/Starlight documentation site surface
for the `innovate` repository.

Baseline decisions:

- `@astrojs/starlight` `0.38.4`
- `starlight-versions` `0.5.4`
- `starlight-links-validator` `0.18.0`
- `@astrojs/starlight-docsearch` `0.6.1`
- `@astrojs/sitemap` `^4.0.0`

Migration policy:

- `parallel-run` while the Sphinx site remains canonical
- Algolia DocSearch as the selected search provider
- canonical URL preservation through redirects during cutover

Key artifacts:

- `astro.config.mjs`
- `starlight.config.mjs`
- `src/content/docs/`
- `../source/_static/astro_starlight/migration_manifest.json`
- `../source/_static/astro_starlight/content_inventory.json`
- `../source/_static/astro_starlight/redirect_inventory.json`
- `../source/_static/astro_starlight/route_coverage.json`
- `../source/_static/astro_starlight/generate_route_coverage.py`
