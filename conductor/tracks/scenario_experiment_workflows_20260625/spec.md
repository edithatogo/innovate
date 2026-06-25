# Scenario Experiment Workflows

## Overview

Add a first-class scenario experiment layer that lets users define, run, compare,
and export reproducible innovation, policy, substitution, and competition
scenarios. This closes the gap between individual model APIs and real-world
decision workflows.

## Functional Requirements

- Provide a stable scenario specification schema for baseline, intervention,
  substitution, competition, and network diffusion workflows.
- Add experiment runner APIs that execute scenario grids and produce comparable
  result envelopes.
- Support reproducible JSON/Arrow-compatible artifacts with diagnostics,
  assumptions, seeds, model metadata, and version metadata.
- Add summary utilities for scenario ranking, incremental effect, uncertainty,
  threshold crossing, and adoption timing.
- Expose the scenario workflow through Python first, with Rust/polyglot payload
  status recorded for each stable shape.
- Add examples and Starlight tutorials for policy, competition, and substitution
  scenario workflows.

## Non-Functional Requirements

- Scenario execution must be deterministic when seeds and inputs are fixed.
- Artifacts must be stable enough for CI, registry evidence, and paper
  reproducibility.
- Optional dependencies must remain optional and fail safely.

## Acceptance Criteria

- Scenario spec models are validated and tested.
- Experiment runner produces stable JSON artifacts.
- Scenario comparisons work across at least policy, competition, and
  substitution examples.
- Starlight docs and examples are present.
- Release-readiness evidence records scenario artifact compatibility.

## Out Of Scope

- Hosted scenario services.
- Interactive UI dashboards.
