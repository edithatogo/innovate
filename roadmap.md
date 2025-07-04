# Roadmap

This document outlines the planned development and future directions for the `heartflow` library, guiding its evolution from an MVP to a full-featured, XLA-ready innovation-diffusion toolkit.

## Phase 1 – MVP & Core Architecture (Weeks 1–4)

**Goal**: Establish the foundational structure and implement core, closed-form diffusion models with basic fitting capabilities.

1.  **Define Scope & Core API**
    *   **Models**: Gompertz, Logistic, Bass (closed-form solutions).
    *   **Estimator Interface**: Implement `fit(t, y)`, `predict(t)`, `score(t, y)`, and `params_` methods.
    *   **Data I/O**: Accept NumPy arrays and pandas Series/DataFrame (with datetime index).

2.  **Project Structure**
    ```
    heartflow/
    ├── diffusion_lib/
    │   ├── __init__.py
    │   ├── backend.py               # abstraction layer (default = NumPyBackend)
    │   ├── backends/
    │   │   ├── numpy_backend.py
    │   │   └── jax_backend.py       # stub for Phase 2
    │   ├── models/
    │   │   ├── base.py              # DiffusionModel ABC
    │   │   ├── bass.py
    │   │   ├── gompertz.py
    │   │   └── logistic.py
    │   │   └── competition.py       # MultiProductDiffusionModel
    │   ├── fitters/
    │   │   ├── scipy_fitter.py
    │   │   └── jax_fitter.py        # stub for Phase 2
    │   ├── utils/
    │   │   ├── preprocessing.py
    │   │   └── metrics.py
    │   └── plots.py                 # matplotlib defaults
    ├── examples/
    │   └── basic_usage.ipynb
    ├── tests/
    │   ├── test_models.py
    │   └── test_fitters.py
    ├── pyproject.toml
    ├── README.md
    └── CONTRIBUTING.md
    ```

3.  **Core Dependencies**
    *   **Runtime**: `numpy`, `pandas`, `scipy` (for nonlinear-LS fitting & ODE if needed).
    *   **Dev & Testing**: `pytest`, `black`, `flake8`.
    *   **Documentation**: `sphinx` (for docs), `mkdocs` or `pdoc`.

4.  **Deliverables**
    *   Basic `fit`/`predict` for each closed-form model.
    *   Automated tests covering edge cases (zero growth, saturated markets, tiny datasets).
    *   Example notebook showing usage on a demo dataset.
    *   Initial implementation of `MultiProductDiffusionModel` for generic competition/substitution.

## Phase 2 – Backend Abstraction & XLA Enablement (Weeks 5–8)

**Goal**: Introduce a flexible backend abstraction to enable high-performance computation with JAX/XLA.

1.  **Finalize the Backend Protocol**
    *   Define a `Backend` Protocol in `backend.py` exposing necessary operations (`exp`, `power`, `ode_solve`, etc.).

2.  **Implement JAX Backend**
    *   Create `jax_backend.py` using `jax.numpy` and `diffrax` for ODE solving.
    *   Wire up `jax_fitter.py` using `jaxopt` for MLE.

3.  **JIT & Vectorization**
    *   Decorate core model functions with `@jax.jit`.
    *   Add `vmap`-based batched fitting for segment/hierarchical use cases.

4.  **Deliverables**
    *   A backend switch mechanism (e.g., `from diffusion_lib.backends import use_backend; use_backend("jax")`).
    *   Benchmarks in `README.md` comparing SciPy vs. JAX on a medium-sized dataset.

## Phase 3 – Advanced Features & Ecosystem (Weeks 9–14)

**Goal**: Expand the library's capabilities with uncertainty quantification, network extensions, and improved visualization.

1.  **Uncertainty Quantification**
    *   Bootstrap wrapper in `fitters/bootstrap_fitter.py`.
    *   Bayesian option via `NumPyro` or `TensorFlow Probability`.

2.  **Network & Agent-Based Extensions**
    *   Optional module `diffusion_lib.networks`.
    *   Simple `simulate_on_graph(G, model, init_seeds, steps)` using `networkx`.

3.  **Forecasting & Scenario Analysis**
    *   Forecast intervals via bootstrap or posterior samples.
    *   Utilities for "what-if" parameter sweeps.

4.  **Visualization & Reporting**
    *   Extend `plots.py` with: Data vs. fit curves, Tornado plots for sensitivity.
    *   Optional `Plotly`/`Panel` dashboard template.

5.  **Packaging & Distribution**
    *   Enable extras in `pyproject.toml` (e.g., `jax = ["jax[cpu]", "jaxopt", "diffrax"]`, `bayes = ["numpyro", "arviz"]`, `network = ["networkx"]`).
    *   Publish v0.1 to PyPI; add CI/CD (GitHub Actions) for linting, tests, and automated doc builds.

## Phase 4 (6–9 months): Competition & Substitution Models

**Goal**: Introduce more sophisticated models for competing innovations and market share dynamics.

*   **Multi-Product Bass & Substitution**: Integrate choice-based models (e.g., Norton–Bass) capturing new adopters and cannibalization. Allow `p` and `q` parameters to depend on relative market shares.
*   **Game-Theoretic Diffusion**: Models where entities choose adoption timing in response to others; compute Nash Equilibria.
*   **Market Share Dynamics**: Lotka–Volterra & Replicator Models for competing technologies/policies. Combine diffusion curves with discrete choice models.

## Phase 5 (9–12 months): Heterogeneity & Segmentation

**Goal**: Account for diverse adopter behaviors and external influences on diffusion.

*   **Latent-Class & Hierarchical Models**: Finite-Mixture Bass to infer adopter segments; Bayesian Hierarchies to pool information across segments/jurisdictions.
*   **Covariate-Driven Parameterization**: Let `p`, `q`, `m` be functions of covariates (e.g., GDP per capita, public awareness) via GLMs or GAMs.
*   **Time-Varying Parameters**: Incorporate piecewise or smoothly evolving `p(t)`/`q(t)` to capture policy shocks or media campaigns.

## Phase 6 (12–18 months): Network & Spatial Extensions

**Goal**: Model diffusion over complex social and geographical structures.

*   **Spatial Diffusion**: Gravity & Spatial-Lag Models; GIS Integration (import shapefiles, map diffusion rates).
*   **Policy Networks**: Model diffusion over directed graphs of influence (e.g., regulatory bodies). Support Watts/Strogatz thresholds for policy cascades.
*   **Agent-Based Simulation Module**: Plug-in where agents interact via network ties and use their own diffusion curves.

## Phase 7 (18–24 months): Causal & Impact Assessment

**Goal**: Enable rigorous causal inference and counterfactual analysis of diffusion processes.

*   **Event-History & Duration Models**: Implement Cox/Aalen models to estimate hazard of policy adoption.
*   **Synthetic Control & DiD Interfaces**: Facilitate integration with causal inference methods to quantify intervention impact.
*   **Counterfactual Scenarios**: Utilities to generate counterfactual adoption trajectories under different baselines.

## Phase 8 (24+ months): Ecosystem & Domain Plugins

**Goal**: Broaden the library's applicability and foster community contributions.

*   **Data Connectors**: Pre-built loaders for public policy indicators (OECD, World Bank).
*   **Domain-Specific Modules**: Health Policy (immunization), Energy Tech (renewables), Technology Standards (5G).
*   **Interactive Dashboards & Reporting**: GIS maps, Sankey diagrams, live scenario sliders.
*   **Community Extensions & Plugin API**: Define an interface for external researchers to contribute models, fitters, or visualizations.

## Cross-Cutting Enablers

*   **Versioning**: Follow Semantic Versioning.
*   **Documentation**: Comprehensive user guide and auto-generated API reference.
*   **Testing**: 80%+ coverage on core modules; performance tests.
*   **Quality**: Pre-commit hooks (`black`, `flake8`, `isort`); code reviews.
*   **R/Julia Interoperability**: Explore `reticulate`/`PyCall.jl` for R/Julia usability, potentially with thin wrapper packages.

This phased plan balances rapid delivery of a usable library with a clear migration path to GPU/TPU-accelerated, XLA-powered performance—while keeping the API stable and the user base growing.
