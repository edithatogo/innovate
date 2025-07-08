# To-Do List

This file tracks the concrete tasks required to execute the project roadmap.

## Phase 1: Foundational Refactoring & Core Diffusion

-   [ ] **Project Restructuring**
    -   [ ] Create new module directories: `src/innovate/diffuse`, `src/innovate/compete`, `src/innovate/substitute`, `src/innovate/hype`, `src/innovate/fail`, `src/innovate/adopt`.
    -   [ ] Move existing model files (Bass, Gompertz, Logistic) into `src/innovate/diffuse/`.
    -   [ ] Move existing fitter and utility files to appropriate new locations.
    -   [ ] Update all internal imports to reflect the new modular structure.
    -   [ ] Run the full test suite to ensure no regressions were introduced during refactoring.
-   [ ] **`innovate.adopt` Module**
    -   [ ] Design an API for adopter classification functions.
    -   [ ] Implement a function to categorize a dataset into Rogers' categories based on adoption timing.
    -   [ ] Write unit tests for the classification logic.
-   [ ] **Documentation**
    -   [ ] Update the main README to explain the new modular architecture.
    -   [ ] Create initial documentation pages for each new module.

## Phase 2: Modeling Competition and Market Dynamics

-   [ ] **`innovate.compete` Module**
    -   [ ] Implement a Lotka-Volterra competition model.
    -   [ ] Create a model class that can simulate multiple S-curves interacting.
    -   [ ] Write tests for competitive scenarios.
-   [ ] **`innovate.substitute` Module**
    -   [ ] Implement a Fisher-Pry substitution model.
    -   [ ] Write tests for technology replacement scenarios.
-   [ ] **`innovate.fail` Module**
    -   [ ] Develop analysis functions to identify conditions of failed diffusion from model outputs (e.g., failure to reach takeoff point).
    -   [ ] Add examples/tutorials for modeling failed adoption.

## Phase 3: Modeling Hype and Sentiment

-   [ ] **`innovate.hype` Module**
    -   [ ] Implement a composite function model (S-curve + Bell curve) for the Hype Cycle.
    -   [ ] Create a modified Bass model with time-varying `p` and `q` parameters.
    -   [ ] Research and select a Python library for solving Delay Differential Equations (DDEs).
    -   [ ] Implement a basic DDE-based hype model.
    -   [ ] Write tests for all hype models.

## Phase 4: Advanced Modeling with Agent-Based Models (ABM)

-   [ ] **ABM Integration**
    -   [ ] Add `mesa` as a core dependency for the ABM module.
    -   [ ] Design the base `InnovationAgent` and `InnovationModel` classes.
-   [ ] **Pre-configured ABM Scenarios**
    -   [ ] Build the competitive diffusion ABM scenario.
    -   [ ] Build the sentiment-driven Hype Cycle ABM scenario.
    -   [ ] Build the disruptive innovation ABM scenario.
-   [ ] **Documentation & Examples**
    -   [ ] Write a comprehensive tutorial on using the ABM framework.
    -   [ ] Document each pre-configured scenario and its parameters.

## Phase 5: Ecosystem, Policy, and Future Directions

-   [ ] **Ecosystem Modeling**
    -   [ ] Design a model for complementary product diffusion.
-   [ ] **Policy Impact**
    -   [ ] Add functionality to models to allow for external shocks or time-varying parameters representing policy interventions.
-   [ ] **Visualization**
    -   [ ] Create a plotting function to easily compare outputs from multiple scenarios.
    -   [ ] Develop a function to visualize network-based diffusion.
