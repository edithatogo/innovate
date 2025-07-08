# Roadmap

This document outlines the planned development for the `innovate` library. The vision is to create a comprehensive, modular, and extensible Python toolkit for modeling the complex dynamics of innovation, competition, and policy diffusion.

## Core Philosophy

The library will be built around a modular architecture, allowing users to combine different models and components to simulate real-world scenarios. It will be grounded in established diffusion theory while providing pathways to advanced techniques like Agent-Based Modeling (ABM).

## Modular Architecture

The library will be organized into the following core modules:

*   `innovate.diffuse`: The foundational module for modeling the adoption of a single innovation.
*   `innovate.compete`: For modeling market share dynamics between two or more competing innovations.
*   `innovate.substitute`: For modeling the replacement of an old technology with a new one.
*   `innovate.hype`: For modeling the Hype Cycle and the influence of public perception and sentiment on adoption.
*   `innovate.fail`: For analyzing the mechanisms and conditions that lead to failed diffusion.
*   `innovate.adopt`: For classifying and analyzing different adopter archetypes (e.g., Rogers' categories).

---

## Phase 1: Foundational Refactoring & Core Diffusion

**Goal**: Establish the new modular architecture and solidify the core diffusion models.

1.  **Restructure Project**: Reorganize the existing codebase into the new modular structure, with the current functionality moving into `innovate.diffuse`.
2.  **Solidify `innovate.diffuse`**:
    *   Ensure core models (Bass, Gompertz, Logistic) are robust.
    *   Refine the `fit`/`predict` API.
    *   Improve documentation and add examples for this core module.
3.  **Develop `innovate.adopt`**:
    *   Implement algorithms to classify adopters from diffusion data based on Rogers' innovation adoption lifecycle (Innovators, Early Adopters, etc.).

## Phase 2: Modeling Competition and Market Dynamics

**Goal**: Introduce models that capture the interaction between multiple innovations.

1.  **Develop `innovate.compete`**:
    *   Implement competitive diffusion models (e.g., Lotka-Volterra).
    *   Add functionality to model multiple interacting S-curves to simulate disruptive innovation scenarios.
2.  **Develop `innovate.substitute`**:
    *   Implement models specifically for technology substitution (e.g., Fisher-Pry).
3.  **Develop `innovate.fail`**:
    *   Create models and analysis tools to understand the conditions for failed adoption, incorporating concepts from competitive and substitution models.

## Phase 3: Modeling Hype and Sentiment

**Goal**: Integrate the dynamics of public perception and hype into the diffusion process.

1.  **Develop `innovate.hype`**:
    *   Implement a mathematical representation of the Hype Cycle (e.g., as a composite function of an S-curve and a hype/attention curve).
    *   Introduce a modified Bass model where parameters (`p`, `q`) can be influenced by a time-varying "hype" or "sentiment" function.
    *   Explore the use of Delay Differential Equations (DDEs) to model the time lags between expectations, performance, and adoption.

## Phase 4: Ecosystem, Policy, and Future Directions

**Goal**: Broaden the library's scope to include more complex real-world factors.

1.  **Ecosystem & Complementary Goods**: Develop models where the adoption of one product is dependent on another (e.g., smartphones and apps).
2.  **Policy & Regulatory Impact**: Add features to simulate the effect of policy interventions (subsidies, mandates) on diffusion rates.
3.  **Path Dependence & Lock-in**: Explore models that demonstrate how early events can lead to long-term market dominance.
4.  **Enhanced Visualization**: Create advanced plotting functions for comparing scenarios, visualizing networks, and animating diffusion processes.

## Phase 5: Advanced Modeling with Agent-Based Models (ABM)

**Goal**: Introduce a powerful new paradigm for bottom-up, emergent modeling.

1.  **Integrate ABM Framework**:
    *   Integrate a library like `Mesa` to serve as the foundation for agent-based simulations.
2.  **Develop Pre-configured ABM Scenarios**:
    *   **Competitive Diffusion**: An ABM for Betamax vs. VHS style competition.
    *   **Hype Cycle**: An ABM with sentiment dynamics to generate emergent hype cycles.
    *   **Disruptive Innovation**: An ABM with incumbent and disruptor firms and heterogeneous customers.
    *   **Policy Diffusion**: An ABM where agents are jurisdictions adopting policies based on network influence.
3.  **Expose ABM Components**: Allow users to define custom agent behaviors, network topologies, and interaction rules.

---
This roadmap provides a clear path forward, balancing the implementation of core, requested features with a vision for a sophisticated and versatile modeling tool.