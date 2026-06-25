# Gap Inventory: Advanced Policy, Competition, and Substitution Modeling

## Overview
This inventory documents gaps across policy, competition, substitution, network, multi-product,
composite, path-dependence, and advanced runtime modules.

## 1. Capabilities Registry Gaps
- **`lock_in` / `path_dependence`**: Missing from `_MODEL_REGISTRY` in `capabilities.py`
  - LockInModel exists at `innovate.path_dependence.lock_in.LockInModel` but has no capability entry
- **`path_dependence` module**: `__init__.py` is empty; no exports defined

## 2. Policy Diffusion Gaps (policy/, models/policy.py)
- **Event-history**: PolicyHazardDiffusionModel has basic event timing but lacks:
  - Staggered rollout duration effects (ramp-up/ramp-down)
  - Spillover diagnostics (regional spillover matrix)
  - Counterfactual scenario comparison at policy level (exists in `causal/counterfactual.py` but not integrated)
  - Uncertainty propagation through policy effects
- **PolicyIntervention**: Only supports BassModel; needs extension to generic models

## 3. Competition Gaps (compete/, dynamics/competition/)
- **Equilibrium/stability checks**: No `equilibrium()` or `stability_analysis()` on LotkaVolterraModel or MultiProductDiffusionModel
- **Cross-elasticity outputs**: No method to compute cross-elasticity from fitted parameters
- **Market-share attraction diagnostics**: MarketShareAttraction in dynamics/ is minimal; no diagnostics payload
- **Phase-plane analysis**: Missing for Lotka-Volterra models

## 4. Substitution Gaps (substitute/)
- **Replacement threshold diagnostics**: No threshold detection on Fisher-Pry or Norton-Bass
- **Scenario comparison**: No method to compare multiple substitution scenarios side-by-side
- **Adoption rate diagnostics**: Missing velocity/acceleration analysis on substitution curves

## 5. Network Diffusion Gaps (models/network.py)
- **Graph-based adoption traces**: NetworkDiffusionModel lacks node-level trace logging
- **Intervention node support**: No API to mark specific nodes as intervention targets
- **Transmissibility diagnostics**: No R0-like metric for network diffusion
- **Ecosystem adapter boundaries**: No integration with ecosystem models

## 6. Multi-Product Gaps (compete/multi_product.py, compete/competition.py)
- **Cross-elasticity computation**: Missing from MultiProductDiffusionModel
- **Equilibrium detection**: No method to detect steady-state market shares
- **Market share projection**: No direct market share (vs adoption) output

## 7. Composite Model Gaps (substitute/composite.py)
- **Interaction diagnostics**: No method to analyze interaction strength between components
- **Scenario comparison**: No cross-scenario comparison payload

## 8. Advanced Runtime Gaps (advanced_runtime.py)
- `compare_policy_scenarios` exists but lacks structured output for multi-scenario comparison
- Missing golden fixtures for schema validation

## 9. Documentation and Test Gaps
- No Starlight docs for gap features
- No model cards for promoted features
- No benchmark cases for competition/substitution scenarios
- No golden fixtures for schema validation
