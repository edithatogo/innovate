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

## Phase 3: Advanced Fitting & Preprocessing (Complete)

**Goal**: Dramatically improve the robustness and accuracy of model fitting and prepare the library for real-world, noisy data.
1.  **Unified Fitting Framework**:
    *   Centralize all fitters into a single `innovate.fitters` module.
    *   Implement a `BayesianFitter` for robust parameter estimation and uncertainty quantification. *(Decision: This will be implemented using `numpyro` to align with the existing JAX backend and resolve stability issues). (Note: This is currently **BLOCKED** due to persistent segmentation fault issues with the underlying Bayesian inference libraries. See `bayesian_fitter_issue.md` for details. Work will proceed on non-Bayesian alternatives like Expectation-Maximization for mixture models.)*
    *   Introduce global optimization strategies (e.g., Differential Evolution) to find better starting parameters and avoid local minima.
2.  **Data Preprocessing Module**:
    *   Create an `innovate.preprocess` module.
    *   Implement robust time-series decomposition methods (e.g., **STL**) to separate trend from seasonality and noise.
    *   Provide a clear pipeline: `preprocess -> fit_on_trend`.
3.  **Model Selection & Diagnostics**:
    *   Add tools for model comparison (AIC, BIC).
    *   Implement robust residual analysis (e.g., plotting autocorrelation) to diagnose model misspecification.

## Phase 4: Functional Core & System Dynamics (Complete)

**Goal**: Refactor the core models to use functional names and introduce a system dynamics module.
1.  **Functional Naming**:
    *   Refactor the core diffusion models (`Bass`, `Gompertz`, `Logistic`) to use functional names (`DualInfluenceGrowth`, `SkewedGrowth`, `SymmetricGrowth`).
2.  **System Dynamics Module (`innovate.dynamics`)**:
    *   Create a new `innovate.dynamics` module to house the core dynamic behaviors.
    *   Implement `GrowthCurve`, `ContagionSpread`, `CompetitiveInteraction`, and `SystemBehavior` abstract base classes.
    *   Implement concrete classes for the core growth models, contagion models (SIR, SIS, SEIR), and competition models (Lotka-Volterra, MarketShareAttraction, ReplicatorDynamics).

## Phase 5: Advanced Diffusion-Competition Models (Complete)

**Goal**: Move beyond simple competition to model more complex market dynamics.
1.  **Generational Substitution Models**:
    *   Implement the **Norton-Bass Model** to explicitly handle the diffusion and substitution of successive product generations.
2.  **Generalized Competition Framework**:
    *   Create a generic `MultiProductDiffusionModel` that can handle an arbitrary number of competing products with a flexible interaction matrix (`Q`). This will serve as a foundation for many competition scenarios.
3.  **Covariate-Driven Models**:
    *   Enhance the core models to allow parameters to be functions of external variables (e.g., price, advertising spend, policy changes).

## Phase 6: High-Performance Backend & Network Science (Complete)

**Goal**: Enable large-scale simulation and more complex network structures.
1.  **JAX/XLA Backend**:
    *   Implement a full `JaxBackend` using `JAX` and high-performance ODE solvers like `Diffrax`.
    *   Ensure the backend can be switched easily by the user.
    *   Provide JIT compilation (`@jax.jit`) and vectorization (`vmap`) for significant performance gains.
2.  **Network Diffusion Enhancements**:
    *   Integrate more deeply with libraries like `NDlib` (Network Diffusion Library).
    -   Implement spatial diffusion models that account for geographic distance (gravity models).

## Phase 7: Heterogeneity & Segmentation (Current Focus)

**Goal**: Model adoption behavior across different population segments.
1.  **Latent-Class & Hierarchical Models**:
    *   Implement finite-mixture models to automatically infer adopter segments.
    *   Develop Bayesian hierarchical models to pool information across segments or jurisdictions.
2.  **Covariate-Driven Parameterization**:
    *   Allow model parameters to be functions of covariates (e.g., GDP per capita, public awareness indices) via GLMs or GAMs.
3.  **Time-Varying Parameters**:
    *   Incorporate piecewise or smoothly evolving parameters to capture policy shocks or media campaigns.

---

## Future Roadmap

The following phases are planned for future releases and are not part of the current development cycle.

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

## Phase 10: Advanced Hype Cycle Modeling

**Goal**: Implement more sophisticated and empirically-grounded models of the Gartner Hype Cycle.
1.  **Superposition Model**:
    *   Implement a hype cycle model based on the superposition of a hype curve (e.g., Gaussian) and an adoption curve (e.g., Logistic or Gompertz).
2.  **Awareness-Knowledge Model**:
    *   Implement a hype cycle model based on the difference between two logistic curves representing "market awareness" and "knowledge acquisition".
3.  **Differential Equation Model**:
    *   Investigate and potentially implement a differential equation-based model for the Gartner Hype Cycle to capture the underlying system dynamics.

## Phase 11: Advanced Substitution and Competition Models

**Goal**: Broaden the scope of substitution and competition modeling.
1.  **Sharif-Kabir Model**:
    *   Implement the Sharif-Kabir model to generalize substitution patterns beyond the symmetric logistic curve.
2.  **Lanchester Models**:
    *   Implement Lanchester's Linear and Square Laws for modeling market share attrition based on competitive effort. The `LanchesterModel` will take initial market shares and effectiveness parameters (`alpha`, `beta`) as input, and simulate the system over time, with an option to select between the linear and square laws.

## Phase 12: Stochastic and Game Theory Models

**Goal**: Introduce stochastic modeling and game theory frameworks.
1.  **Stochastic Diffusion Models (SDEs)**:
    *   Reframe core diffusion models as Stochastic Differential Equations (e.g., adding a Wiener process to the Bass model).
    *   Enable probabilistic forecasting and risk analysis by simulating a distribution of future adoption paths.
2.  **Game Theory Models**:
    *   Implement foundational game theory models for strategic analysis, including Cournot (quantity competition), Bertrand (price competition), and Stackelberg (leader-follower dynamics).
3.  **Advanced Agent-Based Models**:
    *   Implement a `ThresholdModel` as a standard agent type within the ABM framework. This agent adopts based on the fraction of its neighbors who have already adopted, governed by an individual threshold parameter. This will allow for modeling bandwagon effects and social contagion.

## Phase 13: Econometric and Time-Series Models

**Goal**: Integrate traditional econometric models for forecasting and analysis.
1.  **Time-Series Model Integration**:
    *   Provide wrappers or integration for standard time-series models like ARIMA for short-term forecasting.
2.  **Volatility Modeling**:
    *   Implement GARCH models to analyze and forecast the volatility of adoption rates, enabling better risk and supply chain management.

## Phase 14: Advanced Systems Dynamics Constructs

**Goal**: Implement complex, multi-feedback systems dynamics models.
1.  **Multi-Stage Diffusion Models**:
    *   Model the adoption process as a series of stages (e.g., Awareness, Trial, Adoption) with distinct stocks and flows.
2.  **Competitive Strategy Models**:
    *   Develop integrated models that capture the feedback loops between R&D spending, product quality, marketing, market share, and profits.
3.  **Product Portfolio and Lifecycle Models**:
    *   Simulate the dynamics of a firm's entire product portfolio, including R&D pipelines, new product launches, and obsolescence. This will be implemented as a `ProductLifecycleModel` that takes a fitted diffusion model as input and combines it with financial parameters (price, cost, marketing spend) to forecast revenue, profit, and ROI over the product's lifecycle.
4.  **Business and Industry Cycle Models**:
    *   Implement models to simulate and analyze the boom-and-bust cycles inherent in many industries (e.g., semiconductors, real estate).

## Phase 15: Economic and Financial Dynamics

**Goal**: Integrate economic principles directly into the diffusion process.
1.  **Cost-Benefit Diffusion Models**:
    *   Implement an `EconomicDiffusionModel` where the adoption rate is a function of the perceived net benefit. This will allow for modeling how changes in price, product utility, and other costs dynamically influence the diffusion process.
2.  **Dynamic Parameter Models**:
    *   Create a `DynamicBassModel` (and similar models for other diffusion curves) where the innovation and imitation parameters (`p` and `q`) can be arbitrary functions of time or external variables (e.g., advertising spend). This will provide a flexible way to model the impact of marketing and policy drivers.

## Phase 16: Adopter Category Analysis

**Goal**: Provide tools for analyzing and visualizing adopter segments.
1.  **Adopter Category Visualization**:
    *   Add a boolean flag to the plotting functions to allow users to toggle the display of adopter categories (Innovators, Early Adopters, etc.) on the adoption curve plot.
2.  **Category Information Access**:
    *   Implement a function to return the time points or indices corresponding to the boundaries of each adopter category, allowing for programmatic access to information such as the time to reach the Early Majority.

---
This roadmap provides a clear path forward, balancing the implementation of core, requested features with a vision for a sophisticated and versatile modeling tool.