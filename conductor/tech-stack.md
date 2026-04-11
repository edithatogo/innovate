# Technology Stack

## Language
- **Python** (>=3.8)

## Core Libraries
- **NumPy** — Numerical computing and vectorized operations
- **SciPy** — Scientific computing utilities (optimization, interpolation, statistics)
- **Pandas** — Data manipulation with Apache Arrow backend
- **PyArrow** — High-performance columnar data format
- **Statsmodels** — Statistical modeling and hypothesis testing

## Advanced Computation
- **JAX** — GPU/TPU-accelerated numerical computing (optional backend)
- **PyTensor** — Symbolic tensor computation
- **NumPyro** — Probabilistic programming with JAX
- **BlackJAX** — Samplers for Bayesian inference
- **ArviZ** — Exploratory analysis of Bayesian models
- **jitcdde** — Just-in-time compiled delay differential equations
- **diffrax** — Neural differential equations (JAX-based)

## Agent-Based Modeling & Networks
- **Mesa** — Agent-based modeling framework
- **NetworkX** — Graph and network analysis
- **ndlib** — Network diffusion library

## Rupture & Trend Detection
- **ruptures** — Change-point detection
- **pymannkendall** — Mann-Kendall trend tests

## Testing
- **pytest** — Primary test framework
- **pytest-xdist** — Parallel test execution
- **pytest-benchmark** — Performance benchmarking
- **hypothesis** — Property-based testing
- **mutmut** — Mutation testing
- **syrupy** — Snapshot testing

## Code Quality & Linting
- **Ruff** — Fast Python linter and formatter
- **MyPy** — Static type checking
- **Pyright** — Static type checker (Microsoft)
- **Bandit** — Security linting
- **Pylint** — Additional code quality checks
- **pre-commit** — Git hook management

## Documentation
- **Sphinx** — Documentation generator
- **sphinx-rtd-theme** — Read the Docs theme
- **sphinx-autodoc-typehints** — Type hint integration in docs

## Build System
- **setuptools** (>=61.0) — Package building and distribution

## CI/CD
- **GitHub Actions** — Automated testing, linting, and deployment
