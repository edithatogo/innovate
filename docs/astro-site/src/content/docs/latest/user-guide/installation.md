---
title: Installation
description: Detailed installation instructions for Innovate and its backends.
slug: latest/user-guide/installation
---

Innovate supports multiple installation methods and runtime environments.

## System Requirements

* **Python**: 3.14
* **Operating System**: Linux, macOS, Windows
* **RAM**: 4 GB minimum (16 GB recommended for large-scale simulations)

## Package Installation

### From PyPI

The recommended way to install Innovate is from PyPI:

```bash
pip install innovate
```

### Using uv

For faster, reproducible installs:

```bash
uv pip install innovate
```

### From conda-forge

```bash
conda install -c conda-forge innovate
```

### From source

To install the latest development version:

```bash
git clone https://github.com/edithatogo/innovate.git
cd innovate
uv sync
```

## Backend Installation

Innovate uses a multi-backend architecture. The default backend is NumPy.

### JAX Backend (GPU/TPU Accelerated)

```bash
pip install "innovate[jax]"
```

JAX provides hardware acceleration and automatic differentiation, enabling faster fitting of complex models and gradient-based optimisation.

### Numba Backend (CPU JIT)

```bash
pip install "innovate[numba]"
```

Numba compiles numerical kernels at runtime, accelerating tight loops in simulation code without requiring GPU hardware.

### Bayesian Inference Backend

```bash
pip install "innovate[bayesian]"
```

Adds JAX-based MCMC sampling with NumPyro and BlackJAX for full Bayesian inference on diffusion model parameters, providing posterior distributions and credible intervals.

## Verifying the Installation

```python
import innovate
print(innovate.__version__)
```

Confirm the backends are recognised:

```python
from innovate.backend import available_backends
print(available_backends())
```

Expected output:

```
['numpy', 'jax', 'numba', 'bayesian']
```

## Docker

A Docker image with all backends pre-installed is available:

```bash
docker pull ghcr.io/edithatogo/innovate:latest
```

Run a Jupyter notebook:

```bash
docker run -p 8888:8888 ghcr.io/edithatogo/innovate:latest
```
