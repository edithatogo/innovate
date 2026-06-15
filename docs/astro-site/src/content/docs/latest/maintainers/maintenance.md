---
title: Maintenance Policy
description: Evidence-backed maintenance policy for Innovate maintainers.
slug: latest/maintainers/maintenance
---

# Maintenance Policy

Maintenance policy defines the evidence refresh and release readiness cadence.

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
