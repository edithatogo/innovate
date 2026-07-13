---
title: Advanced Policy, Competition, and Substitution Modeling
description: Explore policy diffusion, competition dynamics, and technology substitution models
sidebar:
  order: 6
---

This guide covers the advanced modeling capabilities for policy analysis, competitive dynamics, and technology substitution.

## Policy Diffusion

Policy diffusion models capture how policies spread across regions, with support for:

- **Network spillover effects**: Peer influence through adjacency matrices
- **Intervention targeting**: Mark specific nodes for targeted policy simulation
- **Counterfactual scenarios**: Compare adoption trajectories with and without policy interventions

```python
from innovate.models.network import NetworkDiffusionModel
from innovate.models.contracts import NetworkDiffusionInputs

model = NetworkDiffusionModel(network_inputs=network_data)
model.fit(t, y)
model.set_intervention_nodes([0, 2])  # Target nodes for intervention
predictions = model.predict(t_new)
```

## Competition Models

Multi-product and competitive diffusion with equilibrium analysis:

```python
from innovate.compete.competition import MultiProductDiffusionModel

model = MultiProductDiffusionModel(p=[0.03, 0.04], Q=[[0.4, 0.1], [0.1, 0.5]], m=[100, 200])
model.fit(fitter, t, y)
eq = model.equilibrium()  # Steady-state market shares
elasticity = model.cross_elasticity(t)  # Cross-product sensitivity
```

### Lotka-Volterra Competition

The Lotka-Volterra competition model with equilibrium detection:

```python
from innovate.dynamics.competition import LotkaVolterraCompetition

lvc = LotkaVolterraCompetition()
eq = lvc.equilibrium(carrying_capacity_1=1000, carrying_capacity_2=1000, competition_coeff_12=0.5, competition_coeff_21=0.5)
```

## Substitution Models

Technology substitution analysis with replacement threshold diagnostics:

```python
from innovate.substitute import FisherPryModel, NortonBassModel

fp = FisherPryModel()
fp.fit(fitter, t, y)
diag = fp.threshold_diagnostics()
# Returns: replacement_half_life, takeoff_time, saturation_time, max_adoption_rate_time
```

### Multi-Generation Substitution

The Norton-Bass model extends substitution across multiple technology generations:

```python
nb = NortonBassModel(n_generations=3)
nb.fit(fitter, t, y)
gen_diag = nb.threshold_diagnostics()  # Per-generation thresholds
```

## Path Dependence and Lock-In

The Lock-In model demonstrates network effects and technology lock-in:

```python
from innovate.path_dependence import LockInModel

model = LockInModel()
# Configure intrinsic growth rates and network effect strengths
```

## Capability Registry

All advanced models are registered in the capability system:

```python
from innovate.capabilities import get_model_registry

registry = get_model_registry()
print(registry["lock_in"])  # LockInModel capability metadata
```
