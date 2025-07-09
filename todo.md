# To-Do List

This file tracks the concrete tasks required to execute the project roadmap.

## Phase 1: Foundational Refactoring & Core Diffusion

-   [x] **Project Restructuring**
    -   [x] Create new module directories: `src/innovate/diffuse`, `src/innovate/compete`, `src/innovate/substitute`, `src/innovate/hype`, `src/innovate/fail`, `src/innovate/adopt`.
    -   [x] Move existing model files (Bass, Gompertz, Logistic) into `src/innovate/diffuse/`.
    -   [x] Move existing fitter and utility files to appropriate new locations.
    -   [x] Update all internal imports to reflect the new modular structure.
    -   [x] Run the full test suite to ensure no regressions were introduced during refactoring.
-   [x] **`innovate.adopt` Module**
    -   [x] Design an API for adopter classification functions.
    -   [x] Implement a function to categorize a dataset into Rogers' categories based on adoption timing.
    -   [x] Write unit tests for the classification logic.
-   [x] **Documentation**
    -   [x] Update the main README to explain the new modular architecture.
    -   [x] Create initial documentation pages for each new module.

## Phase 2: Modeling Competition and Market Dynamics

-   [x] **`innovate.compete` Module**
    -   [x] Implement a Lotka-Volterra competition model.
    -   [x] Create a model class that can simulate multiple S-curves interacting.
    -   [x] Write tests for competitive scenarios.
-   [x] **`innovate.substitute` Module**
    -   [x] Implement a Fisher-Pry substitution model.
    -   [x] Write tests for technology replacement scenarios.
-   [x] **`innovate.fail` Module**
    -   [x] Develop analysis functions to identify conditions of failed diffusion from model outputs (e.g., failure to reach takeoff point).
    -   [x] Add examples/tutorials for modeling failed adoption.

## Phase 3: Modeling Hype and Sentiment

-   [x] **`innovate.hype` Module**
    -   [x] Implement a composite function model (S-curve + Bell curve) for the Hype Cycle.
    -   [x] Create a modified Bass model with time-varying `p` and `q` parameters.
    -   [x] Research and select a Python library for solving Delay Differential Equations (DDEs).
    -   [x] Implement a basic DDE-based hype model.
    -   [x] Write tests for all hype models.

## Phase 4: Advanced Modeling with Agent-Based Models (ABM)

-   [x] **ABM Integration**
    -   [x] Add `mesa` as a core dependency for the ABM module.
    -   [x] Design the base `InnovationAgent` and `InnovationModel` classes.
-   [x] **Pre-configured ABM Scenarios**
    -   [x] Build the competitive diffusion ABM scenario.
    -   [x] Build the sentiment-driven Hype Cycle ABM scenario.
    -   [x] Build the disruptive innovation ABM scenario.
-   [x] **Documentation & Examples**
    -   [x] Write a comprehensive tutorial on using the ABM framework.
    -   [x] Document each pre-configured scenario and its parameters.

## Phase 5: Ecosystem, Policy, and Future Directions

-   [x] **Ecosystem Modeling**
    -   [x] Design a model for complementary product diffusion.
-   [x] **Policy Impact**
    -   [x] Add functionality to models to allow for external shocks or time-varying parameters representing policy interventions.
-   [x] **Path Dependence & Lock-in**
    -   [x] Explore models that demonstrate how early events can lead to long-term market dominance.
-   [x] **Visualization**
    -   [x] Create a plotting function to easily compare outputs from multiple scenarios.
    -   [x] Develop a function to visualize network-based diffusion.
