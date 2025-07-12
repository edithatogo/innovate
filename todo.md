# To-Do List

This file tracks the concrete tasks required to execute the project roadmap.

## Phase 1 & 2: Core Models & ABM (Complete)
- [x] All foundational models, the modular architecture, and the initial ABM framework are complete.

## Phase 3: Advanced Fitting & Preprocessing (Current Focus)
-   [x] **Unified `innovate.fitters` Module**
    -   [x] Create the new `src/innovate/fitters` directory.
    -   [x] Move existing fitters from `diffuse/fitters` to the new central location.
    -   [x] Update all necessary imports to reflect the new fitter location.
    -   [x] Run the test suite to ensure no regressions were introduced.
-   [x] **Bayesian Fitter Implementation**
    -   [x] Add `pymc` as a project dependency.
    -   [x] Design the `BayesianFitter` class API.
    -   [x] Implement the fitter, ensuring it can be applied to the library's ODE-based models.
    -   [x] Write unit tests for the `BayesianFitter`. (Note: Tests pass locally but fail with a segfault in the current environment. Deferring further debugging.)
-   [x] **Data Preprocessing & Diagnostics**
    -   [x] Create a new `innovate.preprocess` module.
    -   [x] Implement STL decomposition as a preprocessing step.
    -   [x] Implement robust residual analysis plots (e.g., ACF plots).
-   [x] **Model Selection**
    -   [x] Add AIC/BIC calculation to the model evaluation utilities.
-   [ ] **Documentation**
    -   [ ] Create a comprehensive tutorial for the `BayesianFitter`.
    -   [x] Write a guide on handling seasonal data using the new preprocessing tools.

## Phase 4: Advanced Diffusion-Competition Models (Future)
-   [x] **Implement the Norton-Bass model for generational substitution.**
    -   [x] Create the `NortonBassModel` class structure in `src/innovate/substitute/norton_bass.py`.
    -   [x] Implement the `differential_equation` method.
    -   [x] Implement the `predict` method.
    -   [x] Implement the `initial_guesses` and `bounds` methods.
    -   [x] Implement the `fit` method.
    -   [x] Write unit tests for the `NortonBassModel`.
-   [x] Design and implement the generic `MultiProductDiffusionModel`.
-   [x] Add support for time-varying covariates to core models.
    -   [x] BassModel
    -   [x] GompertzModel
    -   [x] LogisticModel
    -   [x] LotkaVolterraModel
    -   [x] NortonBassModel
        - [x] MultiProductDiffusionModel

## Phase 5 & 6: High-Performance & Causal Inference (Future)
-   [x] Implement a JAX/XLA backend for high-performance computing.
-   [x] Integrate with network science libraries like `NDlib`.
-   [x] Add tools for counterfactual analysis.

