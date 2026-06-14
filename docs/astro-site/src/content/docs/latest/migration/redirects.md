---
title: Redirects
description: Route inventory and forwarder plan for the docs-site cutover.
slug: latest/migration/redirects
---

# Redirects

This page mirrors the redirect inventory that keeps legacy Sphinx URLs
reachable as compatibility aliases for the active Astro/Starlight site.

Route-stability rules:

* Every moved legacy Sphinx page must map to a known Astro route.
* Redirect coverage must stay synchronized with the content inventory.
* Legacy Sphinx URLs remain reachable as compatibility aliases after cutover.

Representative route map:

* `docs/source/index.rst` -> `/`
* `docs/source/innovate.kernel.rst` -> `/core/kernel/`
* `docs/source/binding_publication_ci.rst` -> `/maintainers/publication/`
* `docs/source/release_notes_policy.rst` -> `/maintainers/release-notes/`
* `docs/source/rust_core_roadmap.rst` -> `/operations/rust-core/`
