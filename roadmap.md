# Roadmap

This document outlines the planned development for the `innovate` library. The vision is to create a comprehensive, modular, and extensible Python toolkit for modeling the complex dynamics of innovation, competition, and policy diffusion.

## Core Philosophy

The library will be built around a modular, backend-agnostic architecture. It will provide a simple, intuitive API for common models while allowing for progressive enhancement with high-performance backends (like JAX) and advanced statistical techniques. The core principle is to make robust, reproducible modeling accessible.

---

## Phase 1: Core Models & MVP (Complete)

**Goal**: Establish the core diffusion models and a `scikit-learn`-style API.
-   **Models**: Bass, Gompertz, Logistic, Lotka-Volterra, Fisher-Pry.
-   **API**: A consistent `fit/predict/score` interface.
-   **Backend**: A default backend using NumPy and SciPy.

## Phase 2: Hype, Policy & ABM (Complete)

**Goal**: Broaden the scope to include sentiment dynamics, policy interventions, and agent-based modeling.
-   **Hype Models**: Hype Cycle and DDE-based models.
-   **Policy Module**: Tools to simulate the impact of external shocks.
-   **ABM Framework**: Integration with `Mesa` for bottom-up simulations.

## Phase 3: Advanced Fitting & Preprocessing (Current Focus)

**Goal**: Dramatically improve the robustness and accuracy of model fitting and prepare the library for real-world, noisy data.
1.  **Unified Fitting Framework**:
    *   Centralize all fitters into a single `innovate.fitters` module.
    *   Implement a `BayesianFitter` using `PyMC` for robust parameter estimation and uncertainty quantification.
    *   Introduce global optimization strategies (e.g., Differential Evolution) to find better starting parameters and avoid local minima.
2.  **Data Preprocessing Module**:
    *   Create an `innovate.preprocess` module.
    *   Implement robust time-series decomposition methods (e.g., **STL**) to separate trend from seasonality and noise.
    *   Provide a clear pipeline: `preprocess -> fit_on_trend`.
3.  **Model Selection & Diagnostics**:
    *   Add tools for model comparison (AIC, BIC).
    *   Implement robust residual analysis (e.g., plotting autocorrelation) to diagnose model misspecification.

## Phase 4: Functional Core & System Dynamics

**Goal**: Refactor the core models to use functional names and introduce a system dynamics module.
1.  **Functional Naming**:
    *   Refactor the core diffusion models (`Bass`, `Gompertz`, `Logistic`) to use functional names (`DualInfluenceGrowth`, `SkewedGrowth`, `SymmetricGrowth`).
2.  **System Dynamics Module (`innovate.dynamics`)**:
    *   Create a new `innovate.dynamics` module to house the core dynamic behaviors.
    *   Implement `GrowthCurve`, `ContagionSpread`, `CompetitiveInteraction`, and `SystemBehavior` abstract base classes.
    *   Implement concrete classes for the core growth models, contagion models (SIR, SIS, SEIR), and competition models (Lotka-Volterra, MarketShareAttraction, ReplicatorDynamics).

## Phase 5: Advanced Diffusion-Competition Models

**Goal**: Move beyond simple competition to model more complex market dynamics.
1.  **Generational Substitution Models**:
    *   Implement the **Norton-Bass Model** to explicitly handle the diffusion and substitution of successive product generations.
2.  **Generalized Competition Framework**:
    *   Create a generic `MultiProductDiffusionModel` that can handle an arbitrary number of competing products with a flexible interaction matrix (`Q`). This will serve as a foundation for many competition scenarios.
3.  **Covariate-Driven Models**:
    *   Enhance the core models to allow parameters to be functions of external variables (e.g., price, advertising spend, policy changes).

## Phase 6: High-Performance Backend & Network Science

**Goal**: Enable large-scale simulation and more complex network structures.
1.  **JAX/XLA Backend**:
    *   Implement a full `JaxBackend` using `JAX` and high-performance ODE solvers like `Diffrax`.
    *   Ensure the backend can be switched easily by the user.
    *   Provide JIT compilation (`@jax.jit`) and vectorization (`vmap`) for significant performance gains.
2.  **Network Diffusion Enhancements**:
    *   Integrate more deeply with libraries like `NDlib` (Network Diffusion Library).
    *   Implement spatial diffusion models that account for geographic distance (gravity models).

## Phase 7: Heterogeneity & Segmentation

**Goal**: Model adoption behavior across different population segments.
1.  **Latent-Class & Hierarchical Models**:
    *   Implement finite-mixture models to automatically infer adopter segments.
    *   Develop Bayesian hierarchical models to pool information across segments or jurisdictions.
2.  **Covariate-Driven Parameterization**:
    *   Allow model parameters to be functions of covariates (e.g., GDP per capita, public awareness indices) via GLMs or GAMs.
3.  **Time-Varying Parameters**:
    *   Incorporate piecewise or smoothly evolving parameters to capture policy shocks or media campaigns.

## Phase 8: Causal & Impact Assessment

**Goal**: Bridge the gap between simulation and formal causal impact assessment.
1.  **Event History & Duration Models**:
    *   Integrate survival analysis models (e.g., from the `lifelines` library) to model the "hazard" of policy adoption.
2.  **Counterfactual Analysis**:
    *   Develop tools to simulate "what-if" scenarios and compare them to baseline forecasts, facilitating counterfactual reasoning.
3.  **Integration with Causal Inference Libraries**:
    *   Provide interfaces to libraries like `CausalImpact`, `EconML`, and `DoWhy` to facilitate the use of diffusion models in causal inference pipelines.

## Phase 9: Ecosystem & Domain Plugins

**Goal**: Make the library a central tool for innovation diffusion research.
1.  **Data Connectors**:
    *   Provide pre-built loaders for common datasets (e.g., OECD, World Bank, UN policy indicators).
2.  **Domain-Specific Modules**:
    *   Develop modules for specific domains like health policy, energy tech, and technology standards.
3.  **Interactive Dashboards & Reporting**:
    *   Create templates for interactive dashboards using `Panel` or `Streamlit`.
4.  **Community Extensions & Plugin API**:
    *   Define a plugin interface to allow researchers to contribute new models, fitters, and visualizations.
5.  **Organizational Learning & Ecosystem Dynamics**:
    *   Implement `innovate.organizational_capability` module with `KnowledgeAccumulation` and `AbsorptiveCapacity` models.
    *   Implement `innovate.ecosystems` module with `CoEvolution` and `PlatformMultiSidedGrowth` models.
6.  **Advanced Forecasting & Policy Analysis**:
    *   Implement `innovate.forecasting_utilities` module with `EnsembleForecaster` and `GrowthModelSelector`.
    *   Implement `innovate.policy_tools` module with `PolicyInterventionSimulator` and `PunctuatedEquilibrium` models.
7.  **Product Adoption**:
    *   Implement `innovate.product_adoption` module with `ChurnRetention` and `AttributeBasedChoice` models.

## Phase 10: Performance & Strategy Review

**Goal**: Ensure the library is performant, scalable, and strategically aligned with its goals.
1.  **Computational Efficiency Review**:
    *   Benchmark the performance of the library's core functions.
    *   Identify and address performance bottlenecks.
    *   Investigate opportunities for further optimization, including the use of `pyarrow` and other high-performance libraries.
2.  **Repo Strategy & Structure Review**:
    *   Review the overall strategy and structure of the repository.
    *   Ensure that the library is well-organized, easy to maintain, and aligned with the project's long-term goals.

---
This roadmap provides a clear path forward, balancing the implementation of core, requested features with a vision for a sophisticated and versatile modeling tool.