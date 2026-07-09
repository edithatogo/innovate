---
title: Shared Diagnostics Contract and Uncertainty
description: Standardized fit output across deterministic, bootstrap, and Bayesian fitters.
---

The shared diagnostics contract provides a single shape for model fit metrics,
residual analysis, warnings, and uncertainty summaries across deterministic,
bootstrap, and Bayesian fitters.

## Why this matters

Historically, different fitters exposed different ad hoc fields. The contract
normalizes those outputs so downstream code can compare models, render
visualizations, and surface warnings without special casing each fitter.

## Core types

- `DiagnosticsContract`: standardized fit output for a model.
- `DiagnosticsArtifactPayload`: versioned residual, calibration,
  uncertainty, and model-comparison artifacts for bindings and Arrow
  interchange.
- `UncertaintySummary`: canonical uncertainty report with a `report_type`
  and provenance.
- `DiagnosticsWarning`: structured warning records for unsupported or
  degraded diagnostics.

## Example: inspect a fitted model

```python
import numpy as np
from innovate.diffuse import BassModel
from innovate.fitters import ScipyFitter

t = np.linspace(1, 12, 20)
y = np.linspace(10, 1000, 20)

model = BassModel()
fitter = ScipyFitter()
fitter.fit(model, t, y)

contract = fitter.diagnostics.to_dict()
print(contract["support_level"])
print(contract["uncertainty"]["report_type"])
print(contract["artifacts"]["schema_version"])
```

## Example: compare models consistently

```python
from innovate.utils.model_evaluation import compare_models

comparison = compare_models({"Bass": model}, t, y)
print(comparison[["RMSE", "Diagnostics Support", "Uncertainty Report Type"]])
```

## Example: plot diagnostics from a contract

```python
from innovate.plots.diagnostics import ResidualPlotConfig, plot_residuals

plot_residuals(model, t, y, diagnostics=fitter.diagnostics, config=ResidualPlotConfig(show=False))
```

## Interpretation guidance

- `support_level = supported`: the fitter provided the expected contract.
- `support_level = partial`: diagnostics were available, but some pieces were
  intentionally downgraded or missing.
- `support_level = unsupported`: the model or fitter could not provide a
  valid diagnostics surface, and that state is explicit in the result.

The uncertainty `provenance` field records where the summary came from:
`deterministic`, `bootstrap`, or `bayesian`.

For binding and Arrow consumers, use the versioned artifact payload under
`contract["artifacts"]`. The detailed artifact schema is documented in
`docs/astro-site/src/content/docs/roadmap/diagnostics-uncertainty.md`.
