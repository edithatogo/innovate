# Agent-Based Modeling (ABM) Philosophy

This document outlines the philosophy and planned architecture for the Agent-Based Modeling (ABM) components of the `innovate` library.

## Core Idea

While traditional diffusion models (like Bass or Gompertz) are excellent for understanding aggregate adoption patterns, they treat the market as a homogeneous whole. Agent-Based Modeling allows us to simulate the system from the bottom-up, capturing the heterogeneity and interactions of individual actors (agents) that give rise to the macro-level phenomena we observe.

## Why Use ABM in this Context?

1.  **Heterogeneity**: Model diverse consumers, firms, and policymakers with unique attributes and decision rules.
2.  **Explicit Interactions**: Directly simulate word-of-mouth, social influence, and competitive responses through network structures.
3.  **Emergent Phenomena**: Understand how complex patterns like the Hype Cycle, market-share battles, and technology lock-in emerge from simple individual behaviors.
4.  **"What If" Analysis**: Create a virtual sandbox to test the impact of different strategies, policies, and external shocks.

## Rate Limiting
- To stay within the free tier, do not make excessive API calls. Where possible, use batch-friendly APIs and cache results.

## Planned Agent Types

The library will include pre-configured agents for common scenarios:

*   **Adopter Agents**: Representing consumers with varying levels of risk aversion, social connectivity, and needs (e.g., Innovators, Laggards).
*   **Firm Agents**: Representing incumbent and disruptive companies with strategies for R&D, pricing, and marketing.
*   **Policy Agents**: Representing jurisdictions that can adopt or reject policies based on internal factors and influence from their network neighbors.
*   **Media Agents**: Agents that can influence the sentiment and awareness of other agents in the simulation.
