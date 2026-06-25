---
title: Backends
description: Using Innovate's multi-backend architecture for different
  computation environments.
slug: latest/user-guide/backends
---

Innovate's multi-backend architecture allows you to choose the most appropriate numerical backend for your workload, from lightweight CPU computation to GPU-accelerated simulation.

## Available Backends

| Backend | Hardware | Strengths | Use Case |
|---------|----------|-----------|----------|
| NumPy | CPU | Broad compatibility, no extra deps | Default, quick analysis |
| JAX | CPU/GPU/TPU | Hardware acceleration, autodiff | Large-scale, HPC |
| Numba | CPU (JIT) | Loop acceleration | Simulation-heavy workloads |
| Bayesian | CPU (MCMC) | Full uncertainty quantification | Inference with priors |

## Setting the Backend

Global backend selection:

```python
from innovate import set_backend

# Switch to JAX backend
set_backend('jax')

# All subsequent operations use JAX
result = fit_model(data, model='bass')
```

Per-call backend selection:

```python
result = fit_model(data, model='bass', backend='numba')
```

## NumPy Backend

The default backend. Requires no additional dependencies.

```python
set_backend('numpy')
```

* Best for: quick prototyping, small-to-medium datasets
* Strengths: zero extra dependencies, mature ecosystem integration
* Limitations: CPU-only, no automatic differentiation

## JAX Backend

Hardware-accelerated backend supporting CPU, GPU, and TPU.

```python
set_backend('jax')
```

Installation:

```bash
pip install "innovate[jax]"
```

* Best for: large datasets, HPC clusters, GPU acceleration
* Strengths: XLA compilation, automatic differentiation, hardware portability
* Features: gradient-based optimisation, vectorised simulation

Example with GPU acceleration:

```python
import jax
print(f"Available devices: {jax.devices()}")

set_backend('jax')
result = fit_model(large_data, model='bass')
# Automatically uses GPU if available
```

## Numba Backend

Just-in-time compilation for CPU workloads.

```python
set_backend('numba')
```

Installation:

```bash
pip install "innovate[numba]"
```

* Best for: simulation-heavy agent-based models
* Strengths: significant speedup for tight numerical loops
* Limitations: CPU-only, limited function support

## Bayesian Backend

Full Bayesian inference using the JAX-based NumPyro and BlackJAX stack.

```python
set_backend('bayesian')

result = fit_model(
    data,
    model='bass',
    method='bayesian',
    draws=4000,
    chains=4
)

# Examine posterior distributions
print(result.posterior_summary())
```

Installation:

```bash
pip install "innovate[bayesian]"
```

* Best for: uncertainty quantification, prior incorporation
* Strengths: full posterior distributions, prior specification
* Limitations: computationally intensive, longer run times

## Backend Benchmarks

Comparative performance for fitting a Bass model with 100 data points:

| Backend | Fit Time (ms) | Memory (MB) | Speedup vs NumPy |
|---------|--------------|-------------|------------------|
| NumPy   | 45.2         | 12.3        | 1×               |
| Numba   | 28.7         | 14.1        | 1.6×             |
| JAX (CPU) | 12.4       | 18.6        | 3.6×             |
| JAX (GPU) | 3.1        | 45.2        | 14.6×            |
| Bayesian | 12,400      | 256.0       | 0.004×           |

## Backend-Aware Code

Write code that adapts to the active backend:

```python
from innovate.backend import get_backend

backend = get_backend()
print(f"Active backend: {backend}")

# Backend-agnostic operations work identically
# regardless of the active backend
```
