.. _advanced_diffusion_inference_tutorial:

Advanced Diffusion Inference
============================

This tutorial shows how to use the advanced diffusion model families exposed
through ``innovate.models``.

The advanced workflows are meant for research settings where simple
deterministic diffusion curves are not enough. Typical use cases include:

- comparing pooled and subgroup-specific adoption dynamics
- fitting latent-process residual structure around a baseline diffusion model
- detecting regime changes or structural breaks in adoption data
- generating simulation draws and consistent uncertainty summaries

Backend requirements
--------------------

- The advanced model wrappers run on the standard NumPy backend.
- The regime-switching workflow requires ``ruptures`` for changepoint
  detection. If it is not installed, the rest of the package still works.
- These workflows are intentionally isolated from the base deterministic
  install so optional inference dependencies do not affect core usage.

Example 1: hierarchical diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from innovate.diffuse import BassModel
   from innovate.models import HierarchicalModel

   t = np.arange(1, 21, dtype=float)
   y = np.maximum.accumulate(np.linspace(10, 500, 20))

   model = HierarchicalModel(BassModel(), groups=["north", "south"])
   model.fit(t, {"north": y * 0.9, "south": y * 1.1})

   forecast = model.predict(t)
   draws = model.simulate(t, n_draws=5, random_state=7)
   summary = model.summarize(t)

   print(summary.family)
   print(summary.details["groups"])
   print(draws.shape)

Example 2: latent-process diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from innovate.diffuse import BassModel
   from innovate.models import LatentProcessDiffusionModel

   t = np.arange(1, 31, dtype=float)
   baseline = np.linspace(10, 700, 30)
   observed = np.maximum.accumulate(np.maximum(baseline + np.sin(t) * 15, 0.0))

   model = LatentProcessDiffusionModel(BassModel(), smoothing=0.35)
   model.fit(t, observed)

   forecast = model.predict(t)
   draws = model.simulate(t, n_draws=3, random_state=11)
   summary = model.summarize(t)

   print(summary.details["latent_state_length"])
   print(summary.uncertainty.report_type)
   print(draws.shape)

Example 3: changepoint-aware diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from innovate.diffuse import BassModel
   from innovate.models import RegimeSwitchingDiffusionModel

   t = np.arange(1, 41, dtype=float)
   regime_one = np.linspace(5, 250, 20)
   regime_two = np.linspace(280, 900, 20)
   observed = np.maximum.accumulate(np.concatenate([regime_one, regime_two]))

   model = RegimeSwitchingDiffusionModel(BassModel())
   model.fit(t, observed)

   forecast = model.predict(t)
   summary = model.summarize(t)

   print(summary.details["changepoint_index"])
   print(summary.details["regime_models"])

Interpretation guidance
-----------------------

- Use ``summarize()`` when you need a structured summary that can be consumed
  by reporting code or notebooks.
- Use ``simulate()`` when you need repeated trajectories rather than a single
  point forecast.
- Prefer hierarchical or latent-process models when the main question is
  heterogeneity or smooth residual structure.
- Prefer regime-switching models when a discrete structural break is the
  central hypothesis.
