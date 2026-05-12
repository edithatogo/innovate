# Astro/Starlight migration artifacts

This directory contains the machine-readable state for the docs-site migration.

Files:

- `migration_manifest.json` — transition policy and pinned baseline
- `content_inventory.json` — canonical docs pages and planned Astro routes
- `redirect_inventory.json` — forwarder plan that preserves existing Sphinx
  URLs during parallel-run cutover
- `route_coverage.json` — generated coverage report for implemented and planned
  Astro routes
- `cutover_verification.json` — generated comparison of content and redirect
  inventories
- `link_validation_report.json` — generated route-stability and link-validation
  report
- `generate_route_coverage.py` — generator for the coverage report
- `generate_cutover_verification.py` — generator for the cutover verification
  report
- `generate_link_validation.py` — generator for the link-validation report
