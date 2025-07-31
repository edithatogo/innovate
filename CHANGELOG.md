# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-07-30

### Added
- Implemented a `BayesianFitter` using `PyMC` for robust parameter estimation.
- Added model selection tools for AIC/BIC.
- Added residual analysis plots (ACF/PACF) to `innovate.plots.diagnostics`.
- Implemented a `MultiProductDiffusionModel` for generalized competition scenarios.
- Added support for covariate-driven parameters to all core models.

### Changed
- **Refactored the core of the library into a new `innovate.dynamics` module.**
    - Renamed core diffusion models to have functional names (e.g., `BassModel` is now `DualInfluenceGrowth`).
    - Introduced abstract base classes for `GrowthCurve`, `ContagionSpread`, and `CompetitiveInteraction`.
    - Implemented a full suite of contagion models (SIR, SIS, SEIR) and competition models (Lotka-Volterra, Market Share Attraction, Replicator Dynamics).
- Updated the JAX backend to use `diffrax` for high-performance ODE solving.

## [0.2.1] - 2025-07-12

### Added
- Implemented the Norton-Bass model for generational substitution.
- Implemented a generic Multi-Product Diffusion Model.
- Added a JAX backend for high-performance computing.
- Integrated with NDlib for network-based diffusion modeling.
- Added tools for counterfactual analysis.
- Added example plots for all models to the README.md.

### Changed
- Refactored the `innovate.fitters` module for a more unified structure.
- Refactored the `innovate.utils` module to improve organization and resolve circular imports.
- Updated all license information to be consistent with Apache 2.0.
- Updated Jupyter notebooks to use the new fitter API.

## [0.1.1] - 2025-07-08

### Added
- Added `LICENSE` file (Apache 2.0).
- Added `pyarrow` as a core dependency for efficient data handling with pandas.
- Created `roadmap.md` to outline the future development vision.
- Created `todo.md` for actionable development tasks.
- Created this `CHANGELOG.md`.

### Changed
- Updated project version from 0.1.0 to 0.1.1.
- Republished package to PyPI and created a release on GitHub.
- Prepared for conda publishing by creating a recipe.
