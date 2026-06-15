---
title: Release Maturity Dashboard
description: Evidence-backed release, registry, binding, and Rust ownership status.
slug: latest/operations/release-maturity
---

# Release Maturity Dashboard

This page is generated from machine-readable evidence. It is a status surface,
not a release announcement.

Source dashboard:

- `docs/source/_static/astro_starlight/release_maturity_dashboard.json`

Source artifacts:

- `docs/source/_static/release_readiness_contract.json`
- `docs/source/_static/rust_full_ownership_gate.json`
- `docs/source/_static/registry_submission_inventory.json`
- `docs/source/_static/binding_conformance_inventory.json`

Status summary:

- Release readiness: release-candidate evidence contract is defined.
- Rust ownership: full Rust ownership is not claimed.
- Registry state: not all external registries are accepted.
- Binding conformance: supported bindings are documented against the kernel
  contract.

Guardrails:

- Do not claim all registries accepted until every external registry artifact
  shows accepted or published evidence.
- Do not claim full Rust ownership until the Rust ownership gate allows that
  claim.
