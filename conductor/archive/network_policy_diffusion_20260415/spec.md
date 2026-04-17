# Specification: Network and Policy Diffusion

## Overview

Add modeling support for diffusion processes driven by network structure, spatial adjacency, policy timing, and event-history style hazard formulations so the library can support a broader set of academic and applied adoption studies.

## Functional Requirements

1. Add canonical data structures or adapters for network, spatial, and policy-timing inputs.
2. Implement at least one network-aware diffusion model and one policy or hazard-based diffusion model.
3. Provide a consistent interface for fitting, prediction, simulation, and interpretation across these models.
4. Expose diagnostics or summaries that identify contagion, spillover, or timing effects.
5. Add worked examples for typical empirical settings such as regional adoption or policy diffusion.

## Non-Functional Requirements

1. The implementation must remain compatible with the canonical public API and functional-kernel roadmap.
2. Inputs and outputs must be serializable without Python-specific object assumptions.
3. New capabilities must have clear limits and assumptions documented for users.

## Acceptance Criteria

1. Network and policy models are available through documented canonical imports.
2. Tests cover data validation, fitting behavior, and basic inference outputs.
3. Documentation explains required data shapes and interpretation caveats.
4. Capability metadata reflects support for network and policy workflows.

## Out of Scope

1. Full causal identification frameworks.
2. Interactive visualization tooling.
3. Non-Python language bindings.
