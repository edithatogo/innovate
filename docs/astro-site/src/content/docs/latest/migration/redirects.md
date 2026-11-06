---
title: Redirects
description: Route inventory and forwarder plan for the docs-site cutover.
slug: latest/migration/redirects
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


# Redirects

This page records historical source-to-Astro route mappings for the docs-site
cutover. It is migration evidence, not a second documentation surface.
Former Sphinx URLs remain recorded as historical redirect evidence; Astro/Starlight
routes are the supported documentation URLs.

Route-stability rules:

* Every migrated source page must map to a known Astro route.
* Redirect coverage must stay synchronized with the content inventory.
* Astro/Starlight routes are the supported documentation URLs after cutover.

Representative route map:

* `docs/source/index.rst` -> `/`
* `docs/source/innovate.kernel.rst` -> `/core/kernel/`
* `docs/astro-site/src/content/docs/maintainers/publication.md` -> `/maintainers/publication/`
* `docs/astro-site/src/content/docs/maintainers/release-notes.md` -> `/maintainers/release-notes/`
* `docs/source/rust_core_roadmap.rst` -> `/operations/rust-core/`
