# Innovate

Innovate is a contract-first diffusion modeling library. The current design centers on a stable functional kernel, Arrow interchange, optional backend implementations, and language bindings layered on top of the same core behavior.

The Astro/Starlight site is the active deep documentation. Legacy Sphinx source
is retained only as archival and redirect-reference material. This README is the
short front door.
Markdown prose is checked in CI with Vale to keep the short-form docs clear and
consistent.

## What lives where

- `src/innovate/` contains the Python kernel, fitters, diagnostics, Arrow interchange, and stability layers.
- `bindings/` contains the language bindings and their tests.
- `docs/astro-site/` contains the active Astro/Starlight documentation site.
- `docs/source/` contains legacy Sphinx documentation retained for archival and redirect-reference use.
- `conductor/` contains the track history and archived implementation plans.

## Core ideas

- The functional kernel is the primary surface for fitting, predicting, simulating, summarizing, and diagnosing models.
- Arrow interchange is the stable cross-language data contract.
- Optional backends extend the same contract without changing the public API.
- Bindings are thin layers over the kernel, not separate model implementations.

## Install

```bash
python -m pip install innovate
```

For optional JAX and Bayesian backends:

```bash
python -m pip install "innovate[jax,bayesian]"
```

For contributor setup from a checkout:

```bash
uv sync
```

## Quick start

```python
from innovate import fit_model, predict_model

result = fit_model(data)
forecast = predict_model(result, horizon=12)
```

## Read next

- [Documentation landing page](docs/astro-site/src/content/docs/index.md)
- [Tutorials](docs/astro-site/src/content/docs/tutorials/index.md)
- [Bindings hub](docs/astro-site/src/content/docs/bindings/index.md)
- [Architecture principles](docs/architecture_principles.md)
- [Modernization roadmap](docs/architecture_modernization_roadmap.md)
- [ADR index](docs/adr/index.md)
