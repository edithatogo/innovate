---
title: NDlib Integration for Network-Based Diffusion
description: Legacy NDlib network diffusion (optional legacy-abm extra). Prefer the Kairos simulation adapter for new work.
---

> **Migration note:** Mesa and NDLib are **not** base dependencies. Install
> `pip install innovate[legacy-abm]` before using this tutorial. For new
> simulation work, use the [Kairos Simulation Adapter](/innovate/tutorials/kairos-simulation-adapter/)
> (`innovate.abm.kairos_adapter`) instead.

This tutorial demonstrates how to use the `NdlibInnovationModel` to simulate diffusion on a network.

```python
import networkx as nx
from innovate.abm.ndlib_model import NdlibInnovationModel

# Create a networkx graph
graph = nx.erdos_renyi_graph(n=100, p=0.1)

# Initialize the model
model = NdlibInnovationModel(num_agents=100, graph=graph)

# Run the simulation for 10 steps
for i in range(10):
    model.step()

# Get the status of the nodes
status = model.diffusion_model.status

# Count the number of adopted agents
adopted_agents = [agent for agent, status in status.items() if status == 'Infected']
print(f"Number of adopted agents: {len(adopted_agents)}")
```
