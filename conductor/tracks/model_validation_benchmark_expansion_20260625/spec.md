# Model Validation and Benchmark Expansion

## Overview

Expand validation and benchmark capabilities so model behavior is not only
implemented, but scientifically defensible, reproducible, and comparable across
model families, runtimes, and bindings.

## Functional Requirements

- Add benchmark cases for policy diffusion, competition, substitution,
  network diffusion, multi-product, and causal policy evaluation.
- Add calibration and validation reports with residual diagnostics,
  out-of-sample scoring, sensitivity, and uncertainty coverage.
- Add artifact schemas for benchmark leaderboards and model cards.
- Add reproducibility metadata: dataset version, seed, runtime, dependency
  versions, backend, hardware, and commit.
- Integrate benchmark evidence with release readiness and Rust promotion gates.

## Non-Functional Requirements

- Benchmarks must separate fast CI metadata checks from opt-in timing runs.
- Scientific claims require reproducible artifacts.
- External datasets must have clear licensing and provenance.

## Acceptance Criteria

- Benchmark corpus covers the promoted model families.
- Validation reports are schema-tested and documented.
- Release readiness includes fresh benchmark and validation evidence.
- Starlight docs explain benchmark interpretation and limitations.

## Out Of Scope

- Hosting a public benchmark service.
- Claiming external dataset endorsement.
