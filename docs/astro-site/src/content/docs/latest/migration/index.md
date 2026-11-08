---
title: Migration
description: Cutover policy, inventories, and route-stability notes.
slug: latest/migration
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


# Migration

This section records the cutover policy for the docs-site migration.

* Treat Astro/Starlight as the active documentation stack.
* Keep the content inventory synchronized with Astro routes.
* Keep the redirect inventory as historical route provenance during the cutover.
* Keep historical migration evidence as data artifacts, not as a second docs site.
* See the [redirect route map](/innovate/latest/migration/redirects/) for representative
  source-to-target mappings.
* See the [validation page](/innovate/latest/migration/validation/) for route-stability and
  link-validation checks.
* See the [archive page](/innovate/latest/migration/archive/) for completed Conductor
  evidence.
* See the [migration references page](/innovate/latest/migration/references/) for the
  provenance index.
