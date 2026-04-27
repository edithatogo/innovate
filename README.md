# Innovate

Innovate is a contract-first diffusion modeling library. The current design centers on a stable functional kernel, Arrow interchange, optional backend implementations, and language bindings layered on top of the same core behavior.

The Sphinx site is the canonical deep documentation. This README is the short front door.

## What lives where

- `src/innovate/` contains the Python kernel, fitters, diagnostics, Arrow interchange, and stability layers.
- `bindings/` contains the language bindings and their tests.
- `docs/source/` contains the canonical Sphinx documentation.
- `conductor/` contains the track history and archived implementation plans.

## Core ideas

- The functional kernel is the primary surface for fitting, predicting, simulating, summarizing, and diagnosing models.
- Arrow interchange is the stable cross-language data contract.
- Optional backends extend the same contract without changing the public API.
- Bindings are thin layers over the kernel, not separate model implementations.

## Install

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

- [Documentation landing page](docs/source/index.rst)
- [Tutorials](docs/source/tutorials.rst)
- [Bindings hub](docs/source/bindings.rst)
- [Architecture principles](docs/architecture_principles.md)
- [Modernization roadmap](docs/architecture_modernization_roadmap.md)
- [ADR index](docs/adr/index.md)
