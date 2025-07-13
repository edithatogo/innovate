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

## Phase 4: Advanced Diffusion-Competition Models

**Goal**: Move beyond simple competition to model more complex market dynamics.
1.  **Generational Substitution Models**:
    *   Implement the **Norton-Bass Model** to explicitly handle the diffusion and substitution of successive product generations.
2.  **Generalized Competition Framework**:
    *   Create a generic `MultiProductDiffusionModel` that can handle an arbitrary number of competing products with a flexible interaction matrix (`Q`). This will serve as a foundation for many competition scenarios.
3.  **Covariate-Driven Models**:
    *   Enhance the core models to allow parameters (`p`, `q`, `m`) to be functions of external variables (e.g., price, advertising spend, policy changes).

## Phase 5: High-Performance Backend & Network Science

**Goal**: Enable large-scale simulation and more complex network structures.
1.  **JAX/XLA Backend**:
    *   Implement a full `JaxBackend` using `JAX` and high-performance ODE solvers like `Diffrax`.
    *   Ensure the backend can be switched easily by the user.
    *   Provide JIT compilation (`@jax.jit`) and vectorization (`vmap`) for significant performance gains.
2.  **Network Diffusion Enhancements**:
    *   Integrate more deeply with libraries like `NDlib` (Network Diffusion Library).
    *   Implement spatial diffusion models that account for geographic distance (gravity models).

## Phase 6: Causal Inference & Econometric Models

**Goal**: Bridge the gap between simulation and formal causal impact assessment.
1.  **Event History & Duration Models**:
    *   Integrate survival analysis models (e.g., from the `lifelines` library) to model the "hazard" of adoption.
2.  **Counterfactual Analysis**:
    *   Develop tools to simulate "what-if" scenarios and compare them to baseline forecasts, facilitating counterfactual reasoning.
    .

---
This roadmap provides a clear path forward, balancing the implementation of core, requested features with a vision for a sophisticated and versatile modeling tool.