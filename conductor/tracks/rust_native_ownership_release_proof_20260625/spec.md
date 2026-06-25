# Rust-Native Ownership Release Proof

## Overview

Convert prior Rust-core closure claims into release-grade proof. Every canonical
operation, stable payload shape, and promoted model family must either be
Rust-native, covered by parity tests and benchmarks, or explicitly listed as a
Python-reference exception with release-claim constraints.

## Functional Requirements

- Audit `discover_models`, `fit_model`, `predict_model`, `simulate_model`,
  `summarize_model`, and `diagnose_model` across all stable model families.
- Verify Rust ownership or exception status for diffusion, substitution,
  competition, policy hazard, network diffusion, multi-product, composite, and
  advanced runtime payloads.
- Add parity tests and golden fixtures for every promoted Rust-native operation.
- Update Rust inventory, promotion dossiers, fallback diagnostics, benchmark
  evidence, and release-readiness artifacts.
- Ensure unsupported Rust paths fail closed with structured fallback reasons.

## Non-Functional Requirements

- Rust-native claims require benchmark and conformance evidence.
- Python-reference exceptions must be explicit and cannot be marketed as Rust
  ownership.
- Polyglot bindings must see stable schema-compatible payloads.

## Acceptance Criteria

- Rust ownership inventory is exhaustive and test-validated.
- All stable operations have Rust-native or exception status.
- Golden fixture parity passes across Python and Rust.
- `cargo test`, `cargo check --benches --examples`, and relevant Python tests pass.
- Release docs and evidence do not overclaim Rust ownership.

## Out Of Scope

- Registry publication.
- Starlight visual redesign.
