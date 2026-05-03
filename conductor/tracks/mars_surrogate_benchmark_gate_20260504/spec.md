# Specification: MARS Surrogate Benchmark Gate

## Overview

Benchmark whether `mars` materially improves adoption-curve surrogate workflows
before exposing it as a supported optional backend. This track converts the
ecosystem incubation follow-up "Benchmark whether `mars` improves
adoption-curve surrogate workflows before exposing it as a supported optional
backend" into a narrow evidence-gathering track.

## Roadmap Source

- `docs/ecosystem/module_incubation_strategy.md`
- `specs/ecosystem/README.md`
- Ecosystem incubation follow-up: benchmark `mars` before optional-backend
  promotion

## Functional Requirements

1. Define benchmark scenarios where a MARS surrogate could approximate
   adoption-curve or scenario-response workflows.
2. Compare `mars` against the existing NumPy/SciPy reference path and any
   eligible JAX/XLA-backed alternative where applicable.
3. Record benchmark metrics, correctness tolerances, runtime tier, dependency
   cost, and failure modes.
4. Require benchmark evidence before adding a `mars` optional extra or
   promoting a surrogate adapter.
5. Document the decision outcome as promote, defer, or reject.

## Non-Functional Requirements

1. The base install must not depend on `mars`.
2. The benchmark must use public `mars` APIs only and must not require changes
   to the `mars` core API.
3. Benchmark output must be small, reproducible, and suitable for opt-in CI or
   benchmark workflows.
4. The evaluation must distinguish surrogate benefits from XLA-backed numerical
   kernel acceleration.

## Acceptance Criteria

1. A benchmark plan names scenarios, baselines, tolerances, and promotion
   thresholds.
2. Tests or benchmark metadata checks ensure `mars` remains optional until
   evidence is recorded.
3. Benchmark output records whether gains come from the surrogate, an
   XLA-backed path, or their interaction.
4. Documentation records whether the adapter is promoted, deferred, or
   rejected.

## Out of Scope

1. Adding `mars` as a base dependency.
2. Changing the `mars` package API.
3. Promoting a supported adapter without benchmark evidence.
4. Replacing existing fitting or simulation APIs with surrogate-only workflows.
