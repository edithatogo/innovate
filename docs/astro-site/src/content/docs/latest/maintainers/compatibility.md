---
title: Compatibility Policy
description: Evidence-backed compatibility policy for Innovate maintainers.
slug: latest/maintainers/compatibility
---

# Compatibility Policy

Compatibility policy defines how supported languages and kernel schema commitments are presented.

Evidence:

- `docs/source/_static/astro_starlight/observability_maintenance.json`
- `docs/source/_static/astro_starlight/release_maturity_dashboard.json`
- `docs/source/_static/astro_starlight/production_docs_verification.json`

Operational rules:

- Keep user-facing claims aligned with the machine-readable evidence artifacts.
- Treat external registry or deployment state as pending unless the relevant
  artifact shows accepted, published, or passed evidence.
- Refresh the evidence artifacts before a public release announcement.
