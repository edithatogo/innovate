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
-   [ ] **Bayesian Fitter Implementation**
    -   [x] Add `pymc` as a project dependency.
    -   [x] Design the `BayesianFitter` class API.
    -   [x] Implement the fitter, ensuring it can be applied to the library's ODE-based models.
    -   [ ] Write unit tests for the `BayesianFitter`.
    -   [ ] **Fix segfault issue in the current environment.**
-   [ ] **Data Preprocessing & Diagnostics**
    -   [x] Create a new `innovate.preprocess` module.
    -   [x] Implement STL decomposition as a preprocessing step.
    -   [ ] Implement robust residual analysis plots (e.g., ACF plots).
-   [ ] **Model Selection**
    -   [x] Add AIC/BIC calculation to the model evaluation utilities.
-   [x] **Documentation**
    -   [x] Create a comprehensive tutorial for the `BayesianFitter`.
    -   [x] Write a guide on handling seasonal data using the new preprocessing tools.

## Phase 4: Functional Core & System Dynamics (Future)
-   [ ] **Functional Naming**
    -   [ ] Refactor the core diffusion models (`Bass`, `Gompertz`, `Logistic`) to use functional names (`DualInfluenceGrowth`, `SkewedGrowth`, `SymmetricGrowth`).
-   [ ] **System Dynamics Module (`innovate.dynamics`)**
    -   [ ] Create a new `innovate.dynamics` module to house the core dynamic behaviors.
    -   [ ] Implement `GrowthCurve`, `ContagionSpread`, `CompetitiveInteraction`, and `SystemBehavior` abstract base classes.
    -   [ ] Implement concrete classes for the core growth models, contagion models (SIR, SIS, SEIR), and competition models (Lotka-Volterra, MarketShareAttraction, ReplicatorDynamics).

## Phase 5: Advanced Diffusion-Competition Models (Future)
-   [ ] **Implement the Norton-Bass model for generational substitution.**
-   [ ] Design and implement the generic `MultiProductDiffusionModel`.
-   [ ] Add support for time-varying covariates to core models.

## Phase 6: High-Performance Backend & Network Science (Future)
-   [ ] **JAX/XLA Backend**
    -   [ ] Implement a full `JaxBackend` using `JAX` and high-performance ODE solvers like `Diffrax`.
    -   [ ] Ensure the backend can be switched easily by the user.
    -   [ ] Provide JIT compilation (`@jax.jit`) and vectorization (`vmap`) for significant performance gains.
-   [ ] **Network Diffusion Enhancements**
    -   [ ] Integrate more deeply with libraries like `NDlib` (Network Diffusion Library).
    -   [ ] Implement spatial diffusion models that account for geographic distance (gravity models).

## Phase 7: Heterogeneity & Segmentation (Future)
-   [ ] **Latent-Class & Hierarchical Models**
    -   [ ] Implement finite-mixture models to automatically infer adopter segments.
    -   [ ] Develop Bayesian hierarchical models to pool information across segments or jurisdictions.
-   [ ] **Covariate-Driven Parameterization**
    -   [ ] Allow model parameters to be functions of covariates (e.g., GDP per capita, public awareness indices) via GLMs or GAMs.
-   [ ] **Time-Varying Parameters**
    -   [ ] Incorporate piecewise or smoothly evolving parameters to capture policy shocks or media campaigns.

## Phase 8: Causal & Impact Assessment (Future)
-   [ ] **Event History & Duration Models**
    -   [ ] Integrate survival analysis models (e.g., from the `lifelines` library) to model the "hazard" of policy adoption.
-   [ ] **Counterfactual Analysis**
    -   [ ] Develop tools to simulate "what-if" scenarios and compare them to baseline forecasts, facilitating counterfactual reasoning.
-   [ ] **Integration with Causal Inference Libraries**
    -   [ ] Provide interfaces to libraries like `CausalImpact`, `EconML`, and `DoWhy` to facilitate the use of diffusion models in causal inference pipelines.

## Phase 9: Ecosystem & Domain Plugins (Future)
-   [ ] **Data Connectors**
    -   [ ] Provide pre-built loaders for common datasets (e.g., OECD, World Bank, UN policy indicators).
-   [ ] **Domain-Specific Modules**
    -   [ ] Develop modules for specific domains like health policy, energy tech, and technology standards.
-   [ ] **Interactive Dashboards & Reporting**
    -   [ ] Create templates for interactive dashboards using `Panel` or `Streamlit`.
-   [ ] **Community Extensions & Plugin API**
    -   [ ] Define a plugin interface to allow researchers to contribute new models, fitters, and visualizations.
-   [ ] **Organizational Learning & Ecosystem Dynamics**
    -   [ ] Implement `innovate.organizational_capability` module with `KnowledgeAccumulation` and `AbsorptiveCapacity` models.
    -   [ ] Implement `innovate.ecosystems` module with `CoEvolution` and `PlatformMultiSidedGrowth` models.
-   [ ] **Advanced Forecasting & Policy Analysis**
    -   [ ] Implement `innovate.forecasting_utilities` module with `EnsembleForecaster` and `GrowthModelSelector`.
    -   [ ] Implement `innovate.policy_tools` module with `PolicyInterventionSimulator` and `PunctuatedEquilibrium` models.
-   [ ] **Product Adoption**
    -   [ ] Implement `innovate.product_adoption` module with `ChurnRetention` and `AttributeBasedChoice` models.
