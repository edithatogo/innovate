# Specification: Documentation Version Metadata Synchronization

## Overview

The Python package and Commitizen metadata report version `0.5.0`, while the
Sphinx configuration hard-coded `1.0.0`. This track removes hard-coded docs
version drift by sourcing the Sphinx release from installed package metadata.

## Functional Requirements

- Read the Sphinx `release` and `version` values from package metadata.
- Add a unit test that fails if the Sphinx configuration reintroduces the stale
  hard-coded `1.0.0` value.
- Keep the change scoped to documentation configuration and tests.

## Acceptance Criteria

- `docs/source/conf.py` exposes `release` and `version` equal to
  `importlib.metadata.version("innovate")` when the package is installed.
- Focused docs-version tests pass locally.
- Sphinx docs smoke build continues to run.

## Out of Scope

- Bumping the package version.
- Release workflow changes.
