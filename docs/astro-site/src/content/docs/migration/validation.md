---
title: Validation
description: Route stability and link validation for the Astro/Starlight cutover.
---

# Validation

This page tracks the cutover checks that keep the Astro/Starlight migration
auditable.

- Route coverage is generated from the content inventory and validated against
  the Astro content tree.
- Cutover verification compares the content inventory with the redirect
  inventory.
- Link validation confirms the sidebar routes and internal route links remain
  stable during the parallel-run window.
- route-stability checks ensure the versioned navigation continues to resolve
  the same canonical pages.
- See the [redirect route map](/migration/redirects/), the [archive page](/migration/archive/),
  and the [migration references page](/migration/references/).
