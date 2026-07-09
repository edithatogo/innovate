# Explainability, Sensitivity, and Decision Reports

## Overview

Add explainability and decision-report artifacts so model outputs are usable by
researchers, policy analysts, and decision-makers. The goal is to summarize why
model outcomes change, which assumptions matter, and how robust conclusions are.

## Functional Requirements

- Add sensitivity analysis helpers for parameter perturbation, elasticity,
  scenario assumptions, intervention timing, and threshold outcomes.
- Add explainability summaries for adoption drivers, competition effects,
  substitution thresholds, and policy intervention components.
- Add decision-report artifacts with assumptions, diagnostics, uncertainty,
  limitations, and recommended interpretation language.
- Provide Starlight tutorials and examples for policy, competition, and
  substitution decision reports.
- Ensure reports are JSON/Markdown exportable and release-claim safe.

## Non-Functional Requirements

- Reports must distinguish descriptive, predictive, simulation, and causal
  claims.
- Outputs must be deterministic for fixed inputs.
- Public wording must avoid unsupported recommendations.

## Acceptance Criteria

- Sensitivity and explainability APIs are tested.
- Decision reports export to stable JSON and Markdown.
- Starlight docs explain interpretation boundaries.
- Release evidence records claim-safety checks.

## Out Of Scope

- Automated policy recommendations.
- Legal, clinical, or regulatory advice.
