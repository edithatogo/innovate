# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0](https://github.com/edithatogo/innovate/compare/v0.4.1...v0.5.0) (2026-04-18)


### Features

* **benchmarks:** add benchmark corpus and model cards ([dc042f3](https://github.com/edithatogo/innovate/commit/dc042f333b142738eab8ff56339b63f74f3c8d45))
* **benchmarks:** add stable benchmark suite ([7c990d8](https://github.com/edithatogo/innovate/commit/7c990d86f79f64ce8f0032dbcd3ee921f905e552))
* **bindings-julia:** add Julia kernel binding scaffold ([6d84a63](https://github.com/edithatogo/innovate/commit/6d84a6345c3248c5195a04e8e73dd0189532cae2))
* **bindings-r:** add diagnostics workflow helpers ([c2f857c](https://github.com/edithatogo/innovate/commit/c2f857c799b5723152687e38f75a549d32c7a767))
* **bindings-r:** add packaging guard and example workflow ([d15dbfb](https://github.com/edithatogo/innovate/commit/d15dbfb9336e024adaf9e2ee9f5dbfddc1f8cc37))
* **bindings-r:** implement stable kernel wrappers ([fcb7e91](https://github.com/edithatogo/innovate/commit/fcb7e912860e0acde1eb634a6e96aa19933f0b2a))
* **bindings-r:** scaffold R bindings over functional kernel ([e13f763](https://github.com/edithatogo/innovate/commit/e13f763433b320f15783f51c5768eff735bd3f1e))
* **conductor:** Stabilize optional backends ([ebc5af2](https://github.com/edithatogo/innovate/commit/ebc5af2e7c204da5ec3ea1da0024f3e535cb3019))
* **deps:** Migrate to uv package manager, update docs and Dockerfile ([dd9c657](https://github.com/edithatogo/innovate/commit/dd9c657f90b872709b0bf3b249da52727d171085))
* **diffusion:** add network and policy diffusion contracts ([3d622f3](https://github.com/edithatogo/innovate/commit/3d622f3a2728a509527ccf134924a08f9e1884fe))
* **fitters:** Add multiple optimization methods, diagnostics, and validation ([052ba04](https://github.com/edithatogo/innovate/commit/052ba04bed47c78b62cdf4e6d4badd5b363a2c72))
* **fitters:** Complete Phase 2 - bootstrap CI, parameter correlation, summary ([c390c97](https://github.com/edithatogo/innovate/commit/c390c97c01dadff5340b6ac4887c469466567688))
* **innovate:** add plugin API and stability tiers ([9c1920b](https://github.com/edithatogo/innovate/commit/9c1920bef740864e3f3a8ed47e8c1dd0204b2f96))
* **julia:** add packaging notes and schema drift checks ([4b38790](https://github.com/edithatogo/innovate/commit/4b38790e9f14b3b33dbd79a1873f494e93388176))
* **kernel:** add functional kernel contract surface ([01c7feb](https://github.com/edithatogo/innovate/commit/01c7febf8df76b3cd3d586f73eb8d9d3e4dec3bf))
* **models:** add advanced diffusion workflows ([d91f2db](https://github.com/edithatogo/innovate/commit/d91f2db6bf40c4212cd7d0545230c480b009e607))
* **track:** Complete all phases - examples, docs, QA verification ([45d2de0](https://github.com/edithatogo/innovate/commit/45d2de0227a2de67fc69904c328da1f31dbec062))
* **ts:** add diagnostics workflow example ([841a330](https://github.com/edithatogo/innovate/commit/841a330a591761d3184b2d69176350219a262452))
* **ts:** add stable kernel wrapper bridge ([21b7f6a](https://github.com/edithatogo/innovate/commit/21b7f6af77a189faaa839cc59f64d52899b573bc))
* **ts:** harden package scripts and docs ([4c81379](https://github.com/edithatogo/innovate/commit/4c813792b70ec8688acf0ab1976db85ff39ecae6))
* **ts:** scaffold TypeScript kernel bindings ([f7af800](https://github.com/edithatogo/innovate/commit/f7af8001d70b044df4689ad1b2dc68ba8b373a1f))


### Bug Fixes

* Address linting errors and format code ([923541b](https://github.com/edithatogo/innovate/commit/923541bc4eca0f6dad4863303780de2beabba9a6))
* **ci:** add release drafter template ([38adf27](https://github.com/edithatogo/innovate/commit/38adf271920ca274eda125e07ce0629a5d840ea6))
* **ci:** Fix CI failures — actionlint, bandit, mypy, test collection ([4d780e1](https://github.com/edithatogo/innovate/commit/4d780e1dd99dd30d46d17bea288a24bed5561dc8))
* **ci:** Restore package validation and release config ([e786c42](https://github.com/edithatogo/innovate/commit/e786c42fd0d014a1acd8cb9421bf5f3dddc99e32))
* **conductor:** Apply final review suggestions for track 'Functional Kernel Contract' ([e0c4c72](https://github.com/edithatogo/innovate/commit/e0c4c7248911a840ffdbb787bcdf7c9550225c27))
* **conductor:** Apply review suggestions for track 'Enhance Core Diffusion Models' ([8579ebb](https://github.com/edithatogo/innovate/commit/8579ebb2d23680c6dde3ba715ddab385db80dc0e))
* **deps:** Pin arviz&lt;1.0 for API compatibility, verify tests pass ([a6a4188](https://github.com/edithatogo/innovate/commit/a6a418822154c2234c1ff05181cb9ae366d09591))
* **lint:** Fix all ruff violations in src/ ([fff4ad8](https://github.com/edithatogo/innovate/commit/fff4ad8c280650a6a1613aa2ee32b6d3ab08f0a6))
* **lint:** Fix remaining ruff violations in changed files ([fb17e68](https://github.com/edithatogo/innovate/commit/fb17e68d44967a8283c42d8aad50d528cb138aa8))
* **lint:** Fix ruff B028 stacklevel violations across fitters ([01437b7](https://github.com/edithatogo/innovate/commit/01437b720db65afad4b87f140d5b17f0610a3072))
* **lint:** Fix ruff violations in new code ([3956237](https://github.com/edithatogo/innovate/commit/3956237e8b63f809bc7db0be0581b033be89ca00))
* **tests:** Fix test collection errors and API mismatches ([9a56ad0](https://github.com/edithatogo/innovate/commit/9a56ad03965321e27db88e4700ce2690c5111e49))


### Documentation

* Add GitHub Pages link to repo homepage and pyproject.toml ([5c797ce](https://github.com/edithatogo/innovate/commit/5c797ced1c2f45ffc4ad90452d044b31f6805b3a))
* **benchmarks:** add benchmark workflow docs ([359c788](https://github.com/edithatogo/innovate/commit/359c788e6d9fe0cbc64b68a22d4fb0a437e2b7f1))
* **bindings-r:** document the binding surface ([2ae52c4](https://github.com/edithatogo/innovate/commit/2ae52c41659489a4a32fc668e76292d2334ba430))
* **models:** Document advanced diffusion workflows ([609f3ea](https://github.com/edithatogo/innovate/commit/609f3ea866ece18f30dee18a6dd05cdfc1d8b5fd))

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
