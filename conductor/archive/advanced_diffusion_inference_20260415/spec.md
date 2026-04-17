# Specification: Advanced Diffusion Inference

## Overview

Expand `innovate` beyond baseline deterministic diffusion models by adding hierarchical, state-space, and change-point formulations that support richer uncertainty handling and modern empirical research use cases.

## Functional Requirements

1. Add canonical abstractions for advanced diffusion models that fit within the stable public API.
2. Implement at least one hierarchical diffusion workflow and one state-space or latent-process workflow.
3. Add a change-point or regime-switching diffusion workflow for structural-break analysis.
4. Provide consistent fit, predict, simulate, and summarize interfaces for each advanced model family.
5. Add tests and examples that demonstrate how the advanced models compare with simpler baselines.

## Non-Functional Requirements

1. Optional probabilistic backends must remain isolated from the base install.
2. New model families must integrate with the diagnostics and capability-registry surfaces.
3. APIs must be designed for future exposure through the functional kernel.

## Acceptance Criteria

1. Advanced diffusion models can be imported from canonical package locations.
2. Each new model family has unit tests and end-to-end usage coverage.
3. Advanced models expose uncertainty outputs in a consistent format.
4. Documentation includes clear guidance on backend requirements and intended research use cases.

## Out of Scope

1. Non-Python bindings.
2. GPU-specific optimization work.
3. Production plugin loading infrastructure.
