---
title: Norton-Bass Model for Generational Substitution
description: Simulating diffusion of successive product generations.
---

:::caution[Pre-0.7 reference]
This page records the earlier implementation. The Rust/Polars migration permits temporary feature unavailability; consult the [native migration guide](/innovate/native-migration/) for current APIs and open gates. Earlier language and backend support claims are historical.
:::


This tutorial demonstrates how to use the `NortonBassModel` to simulate the diffusion of successive product generations.

```python
import numpy as np
import matplotlib.pyplot as plt
from innovate.substitute.norton_bass import NortonBassModel

# Initialize the model for 2 generations
model = NortonBassModel(n_generations=2)

# Set the parameters
model.params_ = {
    "p1": 0.03, "q1": 0.2, "m1": 1000,
    "p2": 0.02, "q2": 0.3, "m2": 1500
}

# Generate the time points
t = np.linspace(0, 50, 100)

# Predict the diffusion
y = model.predict(t)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='Generation 1')
plt.plot(t, y[:, 1], label='Generation 2')
plt.title("Norton-Bass Substitution Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
```
