# Ecosystem Contracts

This directory is reserved for health economics and outcomes research (HEOR)
ecosystem-level contracts that connect `innovate` to sibling projects without
importing their internals.

Initial contract targets:

- `lifecourse`: health-intervention adoption and implementation uptake trajectories for
  health-economic scenarios.
- `voiage`: health-intervention adoption and diffusion uncertainty artifacts
  for HEOR VOI workflows.
- `mars`: surrogate/metamodel artifact references and backend metadata for
  optional response-surface workflows.
- HEOML: health-economic extension namespace for uptake, adoption, diffusion,
  and policy-spread artifacts.

Non-goals:

- importing sibling-project internals
- using pickle as a portable contract
- changing the mars core API
- turning `innovate` into a health-economic simulation or VOI engine

Stable contracts should define:

- producer and consumer responsibilities
- schema versions
- required and optional artifacts
- compatibility fixtures
- dependency and optional-extra policy
- diagnostics and provenance fields
- deprecation and migration rules
