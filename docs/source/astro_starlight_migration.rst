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
* ``starlight-links-validator`` ``0.18.0``
* ``@astrojs/starlight-docsearch`` ``0.6.1``
* ``@astrojs/sitemap`` ``^4.0.0``

Scaffold artifacts
------------------

* ``docs/astro-site/package.json``
* ``docs/astro-site/astro.config.mjs``
* ``docs/astro-site/starlight.config.mjs``
* ``docs/astro-site/src/content/docs/``
* ``docs/source/_static/astro_starlight/migration_manifest.json``
* ``docs/source/_static/astro_starlight/content_inventory.json``
* ``docs/source/_static/astro_starlight/redirect_inventory.json``
* ``docs/source/_static/astro_starlight/route_coverage.json``
* ``docs/source/_static/astro_starlight/cutover_verification.json``
* ``docs/source/_static/astro_starlight/generate_route_coverage.py``
* ``docs/source/_static/astro_starlight/generate_cutover_verification.py``

Route-stability policy
----------------------

The migration keeps the current Sphinx URLs reachable until cutover is
complete. The content inventory and redirect inventory are the auditable source
of truth for that agreement.

Next steps
----------

The scaffold is ready for content migration and redirect implementation once
the Astro build is introduced to CI.
