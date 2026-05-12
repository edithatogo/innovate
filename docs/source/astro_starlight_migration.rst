Astro/Starlight documentation site migration
============================================

This page records the implementation scaffold for the future Astro/Starlight
documentation site.

Migration policy
----------------

The migration is a ``parallel-run``:

* the current Sphinx site remains canonical during the transition;
* the Astro/Starlight site is scaffolded in ``docs/astro-site/``;
* the selected search provider is Algolia DocSearch;
* ``@astrojs/sitemap`` is enabled for public indexing;
* the redirect inventory must stay synchronized with the content inventory.

Pinned scaffold baseline
------------------------

* ``@astrojs/starlight`` ``0.38.4``
* ``starlight-versions`` ``0.5.4``
* ``starlight-links-validator`` ``0.24.0``
* ``@astrojs/starlight-docsearch`` ``0.6.1``
* ``astro`` ``^6.0.0``
* ``@astrojs/sitemap`` ``^3.7.2``

Scaffold artifacts
------------------

* ``docs/astro-site/package.json``
* ``docs/astro-site/astro.config.mjs``
* ``docs/astro-site/starlight.config.mjs``
* ``docs/astro-site/package-lock.json``
* ``docs/astro-site/src/content.config.ts``
* ``docs/astro-site/src/content/docs/``
* ``docs/source/_static/astro_starlight/migration_manifest.json``
* ``docs/source/_static/astro_starlight/content_inventory.json``
* ``docs/source/_static/astro_starlight/redirect_inventory.json``
* ``docs/source/_static/astro_starlight/route_coverage.json``
* ``docs/source/_static/astro_starlight/cutover_verification.json``
* ``docs/source/_static/astro_starlight/link_validation_report.json``
* ``docs/source/_static/astro_starlight/generate_route_coverage.py``
* ``docs/source/_static/astro_starlight/generate_cutover_verification.py``
* ``docs/source/_static/astro_starlight/generate_link_validation.py``

Route-stability policy
----------------------

The migration keeps the current Sphinx URLs reachable until cutover is
complete. The content inventory and redirect inventory are the auditable source
of truth for that agreement.

Next steps
----------

Current status is beyond scaffold and into full content migration and final archive.

- Canonical route migration now includes the remaining roadmap, maintainer,
  operations, architecture, and tutorial content from the source inventory.
- Redirect inventory and redirect-to-route mappings are synchronized and machine
  checked through generated cutover verification.
- Link-stability checks now include the expanded sidebar and implementation
  routes added during migration.
- Final review and archive are next when route coverage is complete and the
  migration evidence artifacts remain valid.
