# Causal Policy Evaluation

## Overview

Add causal policy evaluation workflows for analysts who need to estimate,
validate, and communicate the effect of policy interventions on diffusion and
adoption outcomes. This extends the current counterfactual and policy surfaces
from model simulation into evaluation-ready workflows.

## Functional Requirements

- Add policy evaluation contracts for intervention timing, comparator groups,
  staggered rollout, spillovers, covariates, and heterogeneous effects.
- Provide estimation helpers for pre/post comparisons, synthetic/control-style
  summaries, event-study style trajectories, and counterfactual diagnostics.
- Add safeguards for common misuse: missing comparator, post-treatment leakage,
  incompatible time windows, and unsupported causal claims.
- Export stable artifacts with assumptions, estimand, diagnostics, uncertainty,
  and release-claim caveats.
- Integrate policy evaluation artifacts with scenario workflows and model cards.

## Non-Functional Requirements

- Outputs must distinguish simulation from causal evidence.
- Methods must be clearly documented with assumptions and limitations.
- High-risk causal claims must fail closed without sufficient inputs.

## Acceptance Criteria

- Policy evaluation APIs are tested for success and misuse cases.
- Event-study/counterfactual artifacts are schema-validated.
- Docs explain assumptions, limitations, and examples.
- Release evidence prevents unsupported causal claims.

## Out Of Scope

- Replacing specialist econometrics packages.
- Making legal or clinical policy recommendations.
