# Initial Concept

A Python library for simplifying innovation and policy diffusion modeling.

---

# Product Vision

**innovate** is a comprehensive, modular Python library for modeling the diffusion, substitution, competition, and dynamics of innovations, technologies, and policies over time. It serves researchers and practitioners in economics, marketing, public policy, technology forecasting, and epidemiology.

## Core Philosophy

- **Modularity**: Each modeling concern is isolated into focused, composable modules
- **Extensibility**: Clear base classes enable custom model implementations
- **Computational Performance**: Vectorized NumPy operations with optional JAX acceleration
- **Advanced Parameterization**: Support for covariate-driven parameters, time-varying parameters, and mixture models
- **Contract-First Portability**: The stable API, schemas, and functional kernel define the product contract before language-specific bindings
- **Thin Language Bindings**: R, Rust, Julia, TypeScript, Go, C#, and other bindings should expose the same kernel behavior without duplicating model logic
- **Installable Multi-Language Packages**: Binding work should progress toward idiomatic package-manager distribution for each language, including npm, crates.io, R-universe/CRAN, Julia General, versioned Go modules, and NuGet
- **Rust Core Trajectory**: Rust is the strategic long-term core for robust, efficient, portable execution, with Python remaining the primary ergonomic research interface

## Target Users

- **Academic Researchers** in economics, marketing, and public policy
- **Data Scientists** modeling technology adoption and market dynamics
- **Policy Analysts** simulating policy diffusion and impact
- **Forecasters** predicting innovation lifecycles and adoption curves

## Architecture

The library follows a layered architecture. The current Python package provides the primary user-facing API and reference implementation, while the durable product contract is the canonical public API, capability registry, schema layer, and functional kernel. Language bindings must target that contract so behavior stays consistent across Python, R, Rust, Julia, TypeScript, Go, and C#.

Bindings should be treated as product surfaces with CI and packaging obligations, not as examples alone. Each implemented binding needs language-native validation in CI and a publication path through its expected ecosystem package manager.

Over time, performance-critical and portability-critical kernel components should move toward a Rust implementation. Rust should be treated as the strategic core runtime direction, not as another client binding. Python remains the preferred interactive and research-oriented interface, while Rust provides a path to stronger correctness boundaries, packaging portability, and efficient shared execution.

The current Python source layout is:

```
src/innovate/
├── base/           # Abstract base classes for all models
├── backends/       # NumPy/JAX computational backends
├── diffuse/        # Single-innovation adoption curves (Bass, Gompertz, Logistic)
├── substitute/     # Technology replacement models (Fisher-Pry, Norton-Bass)
├── compete/        # Market share competition models
├── hype/           # Gartner Hype Cycle simulation
├── fail/           # Failed adoption analysis
├── adopt/          # Adopter classification
├── dynamics/       # System dynamics (contagion, competition, growth)
├── abm/            # Agent-Based Modeling (Mesa integration)
├── fitters/        # Parameter estimation and curve fitting
├── plots/          # Visualization utilities
├── preprocess/     # Data preprocessing utilities
├── causal/         # Causal inference tools
├── policy/         # Policy analysis tools
├── reduce/         # Dimensionality reduction
├── path_dependence/# Path dependency modeling
├── ecosystem(s)/   # Ecosystem modeling
└── utils/          # Shared utilities
```

## Key Differentiators

1. **Unified Framework**: Combines diffusion models, competition models, and ABM under one roof
2. **Advanced Fitting**: Bayesian (PyMC/NumPyro), JAX-accelerated, and classical optimization
3. **Rich Visualization**: Built-in plotting for diffusion curves, competition dynamics, and diagnostics
4. **Production-Ready**: Comprehensive test suite, type hints, CI/CD, and documentation
5. **Language-Neutral Kernel Contract**: Stable schemas and operations enable consistent bindings across Python, R, Rust, Julia, TypeScript, Go, and C#
6. **Rust-Ready Core Architecture**: The long-term execution core is designed to migrate toward Rust without sacrificing Python usability

## Project Status

- **Version**: 0.5.0
- **License**: Apache 2.0
- **Build System**: setuptools
- **Testing**: pytest with parallel execution, property-based testing, mutation testing
- **Code Quality**: Ruff, MyPy, Pyright, Bandit
- **Documentation**: Sphinx with RTD theme
