---
title: Deprecation Policy
description: Evidence-backed deprecation policy for Innovate maintainers.
slug: latest/maintainers/deprecation
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


# Deprecation Policy

Deprecation policy keeps removals and migrations tied to release readiness evidence.

Evidence:

- `docs/source/_static/astro_starlight/observability_maintenance.json`
- `docs/source/_static/astro_starlight/release_maturity_dashboard.json`
- `docs/source/_static/astro_starlight/production_docs_verification.json`
- `docs/source/_static/astro_starlight/example_validation.json`

Operational rules:

- Keep user-facing claims aligned with the machine-readable evidence artifacts.
- Treat external registry or deployment state as pending unless the relevant
  artifact shows accepted, published, or passed evidence.
- Refresh the evidence artifacts before a public release announcement.
