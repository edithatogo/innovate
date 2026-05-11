---
title: Redirects
description: Route inventory and forwarder plan for the docs-site cutover.
---

# Redirects

This page mirrors the redirect inventory that keeps existing Sphinx URLs
reachable while the Astro/Starlight site runs in parallel.

Route-stability rules:

- Every moved Sphinx page must map to a known Astro route.
- Redirect coverage must stay synchronized with the content inventory.
- Canonical Sphinx URLs remain reachable until cutover completes.

Representative route map:

- `docs/source/index.rst` -> `/`
- `docs/source/innovate.kernel.rst` -> `/core/kernel/`
- `docs/source/binding_publication_ci.rst` -> `/maintainers/publication/`
- `docs/source/release_notes_policy.rst` -> `/maintainers/release-notes/`
- `docs/source/rust_core_roadmap.rst` -> `/operations/rust-core/`
