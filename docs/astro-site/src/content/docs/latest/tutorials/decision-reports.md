---
title: Explainability, Sensitivity, and Decision Reports
description: Claim-safe decision reports with sensitivity analysis and explainability summaries.
---

# Explainability, Sensitivity, and Decision Reports

Decision reports help researchers and policy analysts interpret model outcomes
without automated recommendations. Claims are classified as **descriptive**,
**predictive**, **simulation**, or **causal**, and public wording is fail-closed
against unsupported recommendation language.

## Claim taxonomy

| Claim type | Safe interpretation |
|------------|---------------------|
| descriptive | Patterns under stated data/model only |
| predictive | Projections under fixed assumptions (not causal) |
| simulation | Scenario outcomes under explicit design |
| causal | Requires stated identification assumptions |

```python
from innovate.reports import ClaimRecord, build_decision_report

claim = ClaimRecord(
    claim_type="simulation",
    statement="Under the stated timing scenario, cumulative adoption rises relative to baseline.",
    assumptions=("Fixed market size",),
    limitations=("Simulation is not an empirical trial",),
)
report = build_decision_report(
    title="Policy scenario report",
    workflow="policy",
    claims=(claim,),
    limitations=("Not legal, clinical, or regulatory advice",),
)
```

## Sensitivity helpers

```python
from innovate.reports import (
    ParameterSensitivityInput,
    TimingSensitivityInput,
    ThresholdSensitivityInput,
    parameter_perturbation_summary,
    intervention_timing_summary,
    threshold_sensitivity_summary,
)

def outcome(params):
    return 10.0 + 2.0 * params["beta"] - 0.5 * params.get("t_event", 0.0)

param = parameter_perturbation_summary(
    outcome,
    [ParameterSensitivityInput(name="beta", baseline=1.0, deltas=(-0.1, 0.1))],
)
timing = intervention_timing_summary(
    outcome,
    [TimingSensitivityInput(name="t_event", baseline_time=2.0, offsets=(-1.0, 1.0))],
    context={"beta": 1.0},
)
threshold = threshold_sensitivity_summary(
    [1.0, 2.0, 3.0, 4.0],
    [ThresholdSensitivityInput(name="cut", thresholds=(2.5,))],
)
```

## Explainability summaries

```python
from innovate.reports import (
    adoption_driver_summary,
    competition_effect_summary,
    substitution_threshold_summary,
    policy_component_summary,
)

drivers = adoption_driver_summary(
    {"price": 2.0, "network": 1.0},
    baseline_adoption=10.0,
    scenario_adoption=16.0,
)
competition = competition_effect_summary({"A": 0.4, "B": 0.35}, focal_product="A")
substitution = substitution_threshold_summary([0.1, 0.3, 0.55], thresholds=(0.5,))
policy = policy_component_summary({"subsidy": 0.3, "mandate": 0.1}, total_effect=0.5)
```

## JSON and Markdown export

```python
from innovate.reports import (
    example_policy_report,
    export_report_json,
    export_report_markdown,
)

report = example_policy_report()
print(export_report_json(report)[:200])
print(export_report_markdown(report)[:200])
```

Built-in examples: `example_policy_report`, `example_competition_report`,
`example_substitution_report`.

## Boundaries

- **Out of scope:** automated policy recommendations; legal/clinical/regulatory advice.
- Forbidden phrases (for example “you should implement”, “guaranteed to”) raise errors.
- Sensitivity and explainability blocks are deterministic for fixed inputs.
