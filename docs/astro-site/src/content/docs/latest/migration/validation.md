---
title: Validation
description: Route stability and link validation for the Astro/Starlight cutover.
slug: latest/migration/validation
---

# Validation

This page tracks the cutover checks that keep the Astro/Starlight site
auditable after migration.

* Route coverage is generated from the content inventory and validated against
  the Astro content tree.
* Cutover verification compares the content inventory with the redirect
  inventory.
* Link validation confirms the sidebar routes and internal route links remain
  stable after cutover.
* route-stability checks ensure the versioned navigation continues to resolve
  the same active and compatibility pages.
* `pnpm build && pnpm check` passes for the active site with Python API
  generation and link validation enabled.
* `starlight-versions` remains installed and versioned `latest/` content is
  present, but its active middleware is disabled until it supports Astro 6
  non-doc routes such as `/404`.
* See the [redirect route map](/innovate/latest/migration/redirects/), the [archive page](/innovate/latest/migration/archive/),
  and the [migration references page](/innovate/latest/migration/references/).
