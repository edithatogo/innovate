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

## Target Users

- **Academic Researchers** in economics, marketing, and public policy
- **Data Scientists** modeling technology adoption and market dynamics
- **Policy Analysts** simulating policy diffusion and impact
- **Forecasters** predicting innovation lifecycles and adoption curves

## Architecture

The library follows a layered architecture:

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

## Project Status

- **Version**: 0.5.0
- **License**: Apache 2.0
- **Build System**: setuptools
- **Testing**: pytest with parallel execution, property-based testing, mutation testing
- **Code Quality**: Ruff, MyPy, Pyright, Bandit
- **Documentation**: Sphinx with RTD theme
