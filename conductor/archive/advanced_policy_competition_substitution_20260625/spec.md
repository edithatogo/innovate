# Advanced Policy, Competition, and Substitution Modeling

## Overview

Close related feature gaps beyond infrastructure by advancing the scientific
modeling surface for policy diffusion, competition, substitution, network
diffusion, multi-product diffusion, composite substitution, and cross-language
payload contracts.

## Functional Requirements

- Audit policy, competition, substitution, network, multi-product, composite,
  path-dependence, and advanced runtime model capabilities.
- Identify missing functions, diagnostics, payload schemas, benchmarks, docs,
  and Rust/polyglot binding support.
- Implement missing or immature features where feasible, prioritizing stable
  user value and model correctness.
- Add property-based, regression, benchmark, and golden-fixture tests.
- Promote model-family metadata into capability registries and model cards.
- Update Starlight docs and tutorials for implemented features.

## Recommended Feature Areas

- Policy diffusion: event-history effects, hazard-style timing, spillovers,
  staggered rollout, counterfactual policy scenarios, and uncertainty metadata.
- Competition: multi-product competitive diffusion, Lotka-Volterra parity,
  market-share attraction diagnostics, equilibrium/stability checks, and
  cross-elasticity outputs.
- Substitution: Norton-Bass, Fisher-Pry, composite substitution, replacement
  threshold diagnostics, and scenario comparison payloads.
- Network diffusion: graph-based adoption traces, intervention nodes,
  transmissibility diagnostics, and optional ecosystem adapter boundaries.
- Rust/polyglot payloads: stable JSON/Arrow schemas for each promoted model
  family with explicit Rust-native or Python-reference status.

## Non-Functional Requirements

- Scientific behavior must be tested against invariants and documented
  assumptions.
- Optional dependencies must remain optional and fail safely.
- New model surfaces must be represented in capability registries, docs, and
  release evidence.

## Acceptance Criteria

- A gap inventory exists for all targeted model families.
- Implemented features have tests, docs, examples, model cards, and release
  evidence.
- Stable payloads are schema-validated and binding-compatible.
- Rust-native status or exception status is recorded for each promoted surface.
- Full validation passes or blockers are explicit.

## Out Of Scope

- Publishing papers or external benchmark submissions.
- Claiming Rust-native ownership without evidence.
