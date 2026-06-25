---
title: Plugin API and Stability Tiers
description: Stability tiers, extension manifests, and plugin lifecycle rules.
---

`innovate` treats API stability as an explicit contract instead of a social
convention.

## Stability tiers

The package exposes a normalized three-tier vocabulary through
`innovate.StabilityTier` and `innovate.normalize_stability_tier`:

- `stable` for versioned public contract points
- `provisional` for surfaces that are available but still evolving
- `internal` for implementation details that are not part of the public contract

The capability registries expose a `stability_tier` property so callers can
inspect the normalized tier without relying on string comparisons.

## Extension manifests

Local extensions can be described with `innovate.ExtensionManifest` and
registered with `innovate.register_extension`. The current contract validates:

- a module-style entrypoint string in `module:callable` form
- a declared stability tier
- one or more known extension points

The supported extension points are:

- `model_registry`
- `diagnostics`
- `dataset_provider`
- `serialization_adapter`

## Lifecycle rules

Promotion and deprecation should follow the lifecycle guidance exposed by
`innovate.describe_stability_tier` and the `STABILITY_LIFECYCLE_RULES` table.

In practice:

- stable surfaces may deprecate only with a documented migration window
- provisional surfaces may evolve as the contract settles
- internal surfaces may change without public compatibility guarantees

## Release governance

Release notes and compatibility reviews should name the tier of any new surface
and explain whether it is stable, provisional, or internal-only. That keeps
plugin authors and downstream bindings aligned with semver expectations before
new extension points are promoted.
