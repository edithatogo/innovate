---
title: Fitting Diffusion Models
description: Learn how to fit diffusion models to adoption data using Innovate.
---

Innovate provides a unified interface for fitting a wide range of diffusion models.

## Available Models

Innovate supports the following diffusion model families:

| Model | Description | Parameters |
|-------|-------------|------------|
| Bass | Classic mixed-influence model | p (innovation), q (imitation), m (market potential) |
| Gompertz | Asymmetric S-curve with flexible inflection | a, b, c |
| Logistic | Symmetric S-curve | K (carrying capacity), r (growth rate), t0 (midpoint) |
| Weibull | Flexible hazard-rate model | shape, scale |
| Gumbel | Extreme-value diffusion | location, scale |
| Exponential | Constant-hazard adoption | rate |
| Gamma | Erlang-family adoption | shape, rate |

## Fitting Methods

### Maximum Likelihood Estimation (MLE)

The default fitting method. Finds parameters that maximise the likelihood of observing the data.

```python
from innovate import fit_model

result = fit_model(
    data,
    model='bass',
    method='mle',
    target_col='adoptions',
    time_col='time'
)
```

### Non-linear Least Squares (NLS)

Minimises the sum of squared residuals. Often faster than MLE for large datasets.

```python
result = fit_model(
    data,
    model='gompertz',
    method='nls'
)
```

### Bayesian MCMC

Full Bayesian inference using PyMC. Returns posterior samples for all parameters.

```python
result = fit_model(
    data,
    model='bass',
    method='bayesian',
    draws=2000,
    chains=4
)

# Access posterior samples
posterior = result.posterior
print(posterior['p'].mean())  # Posterior mean of innovation coefficient
```

## Model Comparison

Fit multiple models and compare them using information criteria:

```python
from innovate import fit_model, compare_models

models = ['bass', 'gompertz', 'logistic']
results = {}

for model_name in models:
    results[model_name] = fit_model(data, model=model_name)

comparison = compare_models(results)
print(comparison)
```

Output:
```
          AIC       BIC   LogLik
bass     90.36    91.72   -42.18
gompertz 94.21    95.57   -44.10
logistic 92.88    94.24   -43.44
```

## Custom Initial Parameters

Provide initial parameter guesses to guide the optimiser:

```python
result = fit_model(
    data,
    model='bass',
    method='mle',
    init_params={'p': 0.01, 'q': 0.4, 'm': 500}
)
```

## Parameter Constraints

Bound parameters to reasonable ranges:

```python
result = fit_model(
    data,
    model='bass',
    method='mle',
    bounds={
        'p': (0.0, 1.0),    # Innovation coefficient [0, 1]
        'q': (0.0, 1.0),    # Imitation coefficient [0, 1]
        'm': (100, 10000)   # Market potential
    }
)
```

## Multi-Model Ensembles

Combine multiple models into an ensemble forecast:

```python
from innovate import ensemble

models = ['bass', 'gompertz', 'logistic']
fitted = [fit_model(data, model=m) for m in models]

# Equal-weighted ensemble
ensemble_result = ensemble(fitted, weights='equal')

# Or optimise ensemble weights
ensemble_result = ensemble(fitted, weights='optimise')
```
