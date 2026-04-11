# Technology Stack

## Language
- **Python** (>=3.10) — *3.8/3.9 dropped (EOL)*

## Package Manager & Build
- **uv** — Blazing-fast Python package manager and resolver (replaces pip)
- **setuptools** (>=61.0) — Package building and distribution (managed by uv)
- **uv.lock** — Locked dependency versions for reproducible builds

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

## Linting & Formatting
- **Ruff** — Ultra-fast linter and formatter (replaces Black, isort, flake8, Pylint, vulture, unimport)
  - Rules: F (pyflakes), E/W (pycodestyle), I (isort), B (bugbear), SIM (simplify), UP (pyupgrade), RUF (ruff-specific), C90 (mccabe), N (naming)

## Type Checking
- **ty** — Primary type checker (fastest, by Astral)
- **MyPy** — Secondary type checker (strict mode)

## Testing
- **pytest** — Primary test framework
- **pytest-xdist** — Parallel test execution
- **pytest-benchmark** — Performance benchmarking
- **hypothesis** — Property-based testing
- **mutmut** — Mutation testing
- **syrupy** — Snapshot testing

## Performance Profiling
- **Scalene** — CPU, memory, and GPU profiler with per-line attribution

## Security
- **Bandit** — Security linting for Python code
- **safety** — Dependency vulnerability scanning

## Documentation
- **Sphinx** — Documentation generator
- **sphinx-rtd-theme** — Read the Docs theme
- **sphinx-autodoc-typehints** — Type hint integration in docs
- **MyST-Parser** — Markdown support for Sphinx
- **intersphinx** — Cross-referencing with NumPy, SciPy, Pandas, Mesa docs

## Pre-commit & Git Hooks
- **pre-commit** — Git hook management framework
- **actionlint** — GitHub Actions workflow linter

## CI/CD
- **GitHub Actions** — Automated testing, linting, and deployment
- **Renovate** — Automated dependency updates (replaces Dependabot)
- **release-please** — Conventional-commit-driven releases and changelogs
- **Codecov** — Coverage reporting and PR comments

## Code Quality & Maintenance
- **commitizen** — Conventional commits enforcement + changelog generation
- **pyproject-fmt** — Auto-format pyproject.toml
- **codespell** — Spell checking for code and docs
