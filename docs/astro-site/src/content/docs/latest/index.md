---
title: Innovate Documentation
description: A Python library for simplifying innovation/policy diffusion modelling.
template: splash
hero:
  title: Innovate
  tagline: A Python library for simplifying innovation and policy diffusion modelling.
  actions:
    - text: Get Started
      link: /innovate/latest/user-guide/getting-started/
      icon: right-arrow
      variant: primary
    - text: GitHub
      link: https://github.com/edithatogo/innovate
      icon: external
      variant: minimal
slug: latest
---

import { Card, CardGrid } from '@astrojs/starlight/components';

Innovate provides a unified, contract-first framework for modelling how innovations, policies, and behaviours spread through populations over time. Built on a functional kernel with Arrow-based data interchange, it supports everything from classic Bass diffusion models to agent-based simulations and probabilistic forecasting.

## What Innovate offers

<CardGrid>
  <Card title="Diffusion Modelling" icon="globe">
    Fit, compare, and forecast diffusion models using Bass, Gompertz, Logistic, Weibull, and other S-curve families. Supports multiple fitting methods including MLE, NLS, and Bayesian inference.
  </Card>

  <Card title="Agent-Based Simulation" icon="network">
    Build and run agent-based diffusion models on networks using Mesa integration. Model peer effects, network structure, and heterogeneous adoption thresholds.
  </Card>

  <Card title="Multi-Backend Architecture" icon="layers">
    Choose between NumPy, JAX, or Numba backends for numerical computation. The same API works across all backends, enabling seamless scaling from laptops to HPC clusters.
  </Card>

  <Card title="Arrow Interchange" icon="database">
    All data flows through a stable Arrow-based interchange format, enabling zero-copy interoperability between Python, Rust, R, Julia, and other languages.
  </Card>

  <Card title="Diagnostics & Validation" icon="check-circle">
    Comprehensive model diagnostics including residual analysis, goodness-of-fit tests, change-point detection, and cross-validation.
  </Card>

  <Card title="Language Bindings" icon="code">
    The functional kernel is exposed through bindings for Rust, R, Julia, TypeScript, Go, and C#, all sharing the same core behaviour via the Arrow contract.
  </Card>
</CardGrid>

## Quick Example

```python
import pandas as pd
from innovate import fit_model, predict_model

# Load adoption data (e.g., cumulative adoptions over time)
data = pd.DataFrame({
    'time': range(1, 13),
    'adoptions': [1, 3, 8, 18, 35, 60, 90, 125, 160, 195, 225, 245]
})

# Fit a Bass diffusion model
result = fit_model(data, model='bass', method='mle')

# Forecast 12 periods ahead
forecast = predict_model(result, horizon=12)

print(forecast)
```

## Key Features

* **Contract-first design**: Stable kernel API with backward compatibility guarantees
* **Multiple model families**: Bass, Gompertz, Logistic, Weibull, Gumbel, and more
* **Flexible fitting**: Maximum Likelihood, Non-linear Least Squares, Bayesian MCMC
* **Network effects**: Agent-based simulation with peer influence and network structure
* **Change-point detection**: Automatically detect structural breaks in diffusion patterns
* **Cross-language**: Same models, same results, from Python to Rust to R

## Next Steps

| Topic | Link |
|-------|------|
| Installation guide | [Getting Started](/innovate/latest/user-guide/getting-started/) |
| Core concepts | [Kernel Overview](/innovate/latest/core/kernel/) |
| Full API reference | [Python API](/innovate/latest/api/python/) |
| Migration from Sphinx | [Migration Guide](/innovate/latest/migration/) |
