---
title: Diagnostics and Uncertainty Artifacts
description: Stable diagnostics artifact schema and compatibility contract.
---

# Diagnostics and Uncertainty Artifacts

Innovate exposes diagnostics through two compatible surfaces:

- `DiagnosticsContract` keeps Python-facing metrics, residual analysis, warnings, and uncertainty summaries.
- `DiagnosticsArtifactPayload` adds a versioned, binding-friendly artifact envelope under `diagnostics["artifacts"]`.

## Artifact Contract

The first schema version includes:

- `schema_version`, `model_name`, `support_level`, and `provenance`.
- `backend` and `xla` metadata used by observability and portability tooling.
- `promotion_criteria` and a named `artifacts` block for residual, calibration, uncertainty, and model-comparison payloads.

## Implemented Artifacts

- `residuals`: stable residual diagnostics and summary statistics.
- `uncertainty`: interval-shaped rows with parameter bounds.
- `model_comparison`: metric rows for binding and Arrow consumers.

Migration source:

- `docs/source/diagnostics_uncertainty_artifacts.rst`

