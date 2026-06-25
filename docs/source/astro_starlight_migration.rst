Astro/Starlight documentation site migration
============================================

This page records the implementation scaffold and cutover evidence for the
active Astro/Starlight documentation site.

Migration policy
----------------

The migration status is ``cutover-complete``:

* Astro/Starlight is the active documentation stack in ``docs/astro-site/``;
* legacy Sphinx source is retained as archival and redirect-reference material;
* the selected search provider is Algolia DocSearch;
* ``@astrojs/sitemap`` is enabled for public indexing;
* the redirect inventory must stay synchronized with the content inventory.
* ``pnpm build && pnpm check`` passes for the active site, including
  ``starlight-polyglot`` Python API generation and
  ``starlight-links-validator`` validation.

Pinned scaffold baseline
------------------------

* ``astro`` ``^7.0.2``
* ``@astrojs/starlight`` ``^0.41.0``
* ``@astrojs/markdown-remark`` ``^7.2.0``
* ``starlight-versions`` ``0.9.0``
* ``starlight-links-validator`` ``0.24.1``
* ``@astrojs/starlight-docsearch`` ``0.7.0``
* ``@astrojs/sitemap`` ``^3.7.2``

``starlight-versions`` remains installed and the versioned ``latest/`` content
is present, and the active build validates Astro 7 non-doc routes such as
``/404``.

Scaffold artifacts
------------------

* ``docs/astro-site/package.json``
* ``docs/astro-site/astro.config.mjs``
* ``docs/astro-site/starlight.config.mjs``
* ``docs/astro-site/pnpm-lock.yaml``
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

The migration keeps legacy Sphinx URLs reachable as compatibility aliases. The
content inventory and redirect inventory are the auditable source of truth for
that agreement.

Next steps
----------

Current status is cutover complete with legacy source retained for archive and
redirect evidence.

- Canonical route migration now includes the remaining roadmap, maintainer,
  operations, architecture, and tutorial content from the source inventory.
- Redirect inventory and redirect-to-route mappings are synchronized and machine
  checked through generated cutover verification.
- Link-stability checks now include the expanded sidebar and implementation
  routes added during migration.
- Final review and archive evidence remain valid when route coverage and link
  validation artifacts stay synchronized.
