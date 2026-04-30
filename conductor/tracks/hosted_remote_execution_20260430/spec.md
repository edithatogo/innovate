# Specification: Hosted Services and Remote Execution

## Overview

Define hosted service and remote execution layers that reuse the functional kernel contract rather than inventing a second API. This track turns the roadmap item "hosted services or remote execution layers" into a staged design and implementation path.

## Roadmap Source

- `docs/architecture_modernization_roadmap.md`
- Deferred work item: "hosted services or remote execution layers"

## Functional Requirements

1. Define service boundaries for remote execution, including request, response, error, provenance, and version fields.
2. Reuse the functional kernel schemas and Arrow-compatible interchange wherever possible.
3. Define authentication, authorization, tenant isolation, and data-retention expectations before any hosted implementation.
4. Define observability requirements for structured logs, traces, metrics, and request correlation.
5. Prototype a minimal local or test service only after the contract and threat model are documented.
6. Document which operations are eligible for remote execution and which remain local-only.
7. Include optional accelerator placement rules so hosted execution can report whether a run used NumPy/SciPy, JAX/XLA, or Rust-native execution.

## Non-Functional Requirements

1. Remote execution must not weaken schema compatibility, versioning, or reproducibility guarantees.
2. Sensitive inputs and outputs must have explicit handling rules before hosted deployment.
3. The first implementation slice must be testable without provisioning production infrastructure.
4. Failure modes must be structured and language binding friendly.
5. Remote execution must not require clients to understand XLA internals to consume results.

## Acceptance Criteria

1. A remote execution contract and threat model exist.
2. Observability, authentication, and data-handling expectations are documented.
3. A minimal testable service or adapter slice demonstrates the contract without production deployment.
4. CI validates request/response compatibility and structured error behavior.

## Out of Scope

1. Deploying a production hosted service in the first track.
2. Replacing the local Python API or functional kernel.
3. Supporting arbitrary remote code execution.
