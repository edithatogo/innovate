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
- explicit promotion stages from documented to experimental to supported
- the smoke CI, Renovate, security, and documentation gates needed for each
  optional adapter

Expected artifact groups include:

- adoption curves and uptake trajectories
- policy-spread traces and network diffusion traces
- diagnostics, calibration outputs, and provenance records
- Arrow or Parquet tabular payloads plus JSON manifests and schemas

HEOML alignment should treat `innovate` artifacts as the portable base layer
and only add extension metadata when a health-economic bundle requires a shared
namespace or cross-repo wrapper.

Namespace guidance:

- keep `innovate` artifact names for repo-local consumers
- use `heoml.extensions.innovate` for cross-repo HEOR bundles
- preserve the same underlying tabular payload across both representations

Dependency policy:

- keep the base install free of sibling-project dependencies
- expose adapters only through optional extras or equivalent explicit flags
- require deterministic fixtures and smoke CI before promotion
- require a compatibility matrix before an adapter is marked supported

Current scaffolds:

- [process/README.md](./process/README.md) for the HEOR process-mining outline
  and PM4Py ecosystem-only contract
