# Technology Stack

## Language
- **Python** (>=3.10) — *3.8/3.9 dropped (EOL)*
- **R** — Thin user-facing binding over the stable functional kernel contract
- **Rust** — Current binding target and strategic long-term core runtime for robust, efficient, portable kernel execution
- **Julia** — Thin user-facing binding over the stable functional kernel contract
- **TypeScript** — Thin user-facing binding over the stable functional kernel contract
- **Go** — Thin user-facing binding over the stable functional kernel contract
- **C#** — Planned thin binding target over the stable functional kernel contract

## Runtime Strategy
- **Python-first API stabilization** — The canonical Python public API, capability registry, schemas, and functional kernel define the stable product contract before additional language expansion.
- **Thin binding policy** — R, Rust, Julia, TypeScript, Go, and C# bindings must call or mirror the shared kernel contract. They should not fork or reimplement model behavior independently.
- **Rust core trajectory** — Rust is the preferred long-term implementation language for performance-critical and portability-critical kernel components. Rust work should evolve from binding/client coverage toward shared core execution while preserving Python ergonomics.
- **Reference semantics** — Python/NumPy/SciPy remains the reference correctness path until Rust components are promoted behind the same contract and validated by parity tests.

## Package Manager & Build
- **uv** — Blazing-fast Python package manager and resolver (replaces pip)
- **setuptools** (>=61.0) — Package building and distribution (managed by uv)
- **uv.lock** — Locked dependency versions for reproducible builds
- **Node.js** (>=22) — Runtime for the TypeScript binding package and test harness
- **npm** — Package manager and publication target for the TypeScript binding workspace
- **cargo / crates.io** — Rust build tooling and publication target for Rust binding and future core components
- **R package tooling / R-universe / CRAN** — DESCRIPTION/NAMESPACE-based package structure, near-term R-universe publication path, and longer-term CRAN target
- **Julia package tooling / Julia General registry** — Project-based package structure and eventual Julia registry publication path
- **Go toolchain / versioned Go modules** — Go module tooling and release-tag publication path for Go binding validation
- **.NET SDK / NuGet** — Planned toolchain and package publication target for C# binding validation

## Core Libraries
- **NumPy** — Reference numerical backend and Array API baseline
- **SciPy** — Scientific computing utilities (optimization, interpolation, statistics) with emerging Array API support where practical
- **Pandas** — Primary user-facing DataFrame API in Python
- **PyArrow** — Columnar types and interchange layer for pandas integration and future bindings
- **Statsmodels** — Statistical modeling and hypothesis testing

## Portability and Interchange Standards
- **Python Array API standard** — Numerical portability target for durable kernel semantics
- **Apache Arrow** — Durable columnar and tabular interchange boundary for kernel payloads and future non-Python bindings
- **Polars** — Optional ETL and benchmark-ingestion engine; not part of the stable public API contract

## Advanced Computation
- **JAX** — Accelerator backend for fitting, simulation, and inference (optional, not public ABI)
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
- **Multi-language binding CI** — Dedicated GitHub Actions jobs for Rust, TypeScript, Go, Julia, and R binding validation
- **Binding publication workflow** — Release-gated package checks and publication hooks for npm, crates.io, R, Julia, Go modules, and planned NuGet support

## Code Quality & Maintenance
- **commitizen** — Conventional commits enforcement + changelog generation
- **pyproject-fmt** — Auto-format pyproject.toml
- **codespell** — Spell checking for code and docs
