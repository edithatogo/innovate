---
title: Python API Reference
description: Complete API reference for the Innovate Python library.
---

This page documents the public API of the `innovate` Python package.

---

## Core Functions

### `fit_model(data, model='bass', method='mle', **kwargs)`

Fit a diffusion model to adoption data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Adoption data with time and value columns |
| `model` | `str` | `'bass'` | Model family: `'bass'`, `'gompertz'`, `'logistic'`, `'weibull'`, `'gumbel'`, `'exponential'`, `'gamma'` |
| `method` | `str` | `'mle'` | Fitting method: `'mle'`, `'nls'`, `'bayesian'` |
| `target_col` | `str` | `'adoptions'` | Column name for adoption values |
| `time_col` | `str` | `'time'` | Column name for time index |
| `backend` | `str` | `None` | Numerical backend: `'numpy'`, `'jax'`, `'numba'`, `'bayesian'` |
| `init_params` | `dict` | `None` | Initial parameter guesses |
| `bounds` | `dict` | `None` | Parameter bounds per model |
| `draws` | `int` | `2000` | (Bayesian only) MCMC draws per chain |
| `chains` | `int` | `4` | (Bayesian only) Number of MCMC chains |

**Returns:** `FitResult` object with `params`, `standard_errors`, `log_likelihood`, `aic`, `bic`, `posterior`, `model`, `method`.

---

### `predict_model(result, horizon=12, **kwargs)`

Generate forecasts from a fitted model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FitResult` | required | Fitted model from `fit_model` |
| `horizon` | `int` | `12` | Number of periods to forecast |
| `type` | `str` | `'cumulative'` | Output type: `'cumulative'`, `'new'`, `'rate'` |
| `confidence_level` | `float` | `None` | Confidence level for intervals (e.g., `0.95`) |
| `n_simulations` | `int` | `1000` | Bootstrap simulations for intervals |
| `covariates` | `pd.DataFrame` | `None` | External covariates for forecast |

**Returns:** `PredictionResult` with `time`, `mean`, `confidence_lower`, `confidence_upper`, `type`.

---

### `summarise_model(result, forecast=None)`

Generate a human-readable summary of a fitted model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FitResult` | required | Fitted model result |
| `forecast` | `PredictionResult` | `None` | Optional forecast to include |

**Returns:** `str` — Formatted summary table.

---

### `compare_models(results)`

Compare multiple fitted models using information criteria.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `dict[str, FitResult]` | required | Dict mapping model names to fit results |

**Returns:** `pd.DataFrame` with AIC, BIC, and log-likelihood for each model.

---

### `ensemble(models, weights='equal')`

Combine multiple models into an ensemble forecast.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[FitResult]` | required | List of fitted models |
| `weights` | `str | list` | `'equal'` | Weighting: `'equal'`, `'optimise'`, or array |

**Returns:** `EnsembleResult` with combined predictions.

---

### `peak_time(result)`

Calculate the peak adoption time for a fitted model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FitResult` | required | Fitted model result |

**Returns:** `float` — Time period of peak adoption.

---

### `set_backend(name)`

Set the global numerical backend.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Backend name: `'numpy'`, `'jax'`, `'numba'`, `'bayesian'` |

---

## Model Families

### Bass Model

`from innovate.models import BassModel`

The Bass diffusion model describes adoption through innovation (external, `p`) and imitation (internal, `q`).

```
f(t) = (p + q * F(t)) * (1 - F(t))
```

- `p` = coefficient of innovation (external influence)
- `q` = coefficient of imitation (internal influence)
- `m` = market potential (total adopters)
- `F(t)` = cumulative adoption fraction at time `t`

### Gompertz Model

`from innovate.models import GompertzModel`

Asymmetric S-curve with flexible inflection point.

```
f(t) = a * exp(-b * exp(-c * t))
```

### Logistic Model

`from innovate.models import LogisticModel`

Symmetric S-curve fixed at 50% carrying capacity.

```
f(t) = K / (1 + exp(-r * (t - t0)))
```

### Weibull Model

`from innovate.models import WeibullModel`

Flexible hazard-rate model.

```
f(t) = (k / λ) * (t / λ)^(k-1) * exp(-(t / λ)^k)
```

---

## Diagnostics

### `from innovate.diagnostics import residual_analysis`

Performs residual diagnostics: residual vs fitted, Q-Q plot, ACF, Ljung-Box test.

### `from innovate.diagnostics import cross_validate`

```python
cross_validate(data, model='bass', k=5)
```

Time-series cross-validation with `k` folds.

### `from innovate.diagnostics import change_point_detection`

Detect structural breaks in diffusion patterns using the PELT algorithm.

---

## Agent-Based Simulation

### `from innovate.abm import ABMSimulation`

Build and run agent-based diffusion models on networks.

```python
from innovate.abm import ABMSimulation
from innovate.abm.networks import small_world

sim = ABMSimulation(
    network=small_world(n=1000, k=6, p=0.1),
    adoption_threshold='threshold',
    threshold_params={'mean': 0.5, 'std': 0.15},
    influence='peer',
    max_time=100
)
results = sim.run()
```

---

## Arrow Interchange

### `from innovate import to_arrow, from_arrow`

Convert between Innovate data and Apache Arrow format.

```python
arrow_table = to_arrow(result)
result_restored = from_arrow(arrow_table)
```

The Arrow interchange provides a stable, cross-language data contract used by all language bindings (Rust, R, Julia, TypeScript, Go, C#).

---

## Plugin System

### `from innovate.plugins import register_model, list_plugins`

Extend Innovate with custom model families.

```python
from innovate.plugins import register_model

@register_model(name='my_custom_model')
class MyCustomModel:
    """Custom diffusion model implementation."""
    ...
```