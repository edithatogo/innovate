# To-Do List

This file tracks the concrete tasks required to execute the project roadmap. It has been updated to reflect the current state of the codebase.

## High Priority Tasks

- [ ] **Define HEOR module naming brainstorm**
    -   Keep `calibrate`, `evidence`, `process`, `report`, `registry`, `workflow`, `quality`, `engines`, and `heoml` as the short list.
    -   Keep PM4Py in the ecosystem-only process-mining bucket.
    -   Require CLI support for every future module and decide whether MCP is useful case by case.

- [ ] **Define ecosystem module incubation policy**
    -   Document the `innovate` role alongside `lifecourse`, `voiage`, `mars`, and HEOML.
    -   Keep the ecosystem scope focused on health economics and outcomes research.
    -   Define adoption/diffusion artifacts that can feed health-economic scenarios and VOI workflows.
    -   Reserve HEOML extension alignment for uptake, adoption, diffusion, and policy-spread artifacts.
    -   Keep `mars` as a fixed-API optional surrogate/metamodel backend.
    -   Require optional extras, smoke CI, and compatibility matrices before supported status.
    -   Promote adapters from documented to experimental to supported only after release policy is clear.

- [ ] **Refactor Bayesian Fitter with NumPyro (Blocked)**
    -   **Reason**: Unable to resolve dependency conflicts between `jax`, `numpyro`, and `scipy`.
    -   [ ] Add `numpyro` as a dependency to `pyproject.toml`.
    -   [ ] Remove the existing `pymc`-based `BayesianFitter`.
    -   [ ] Implement a new `NumpyroFitter` that leverages the JAX backend and `diffrax`.
    -   [ ] Write comprehensive unit tests for the `NumpyroFitter`.

## Documentation

- [x] **Synchronize Core Documentation**
    -   [x] Update `roadmap.md` to accurately reflect the completed status of Phases 4, 5, and 6.
    -   [x] Draft a new `v0.3.0` entry for `CHANGELOG.md` to capture all features implemented since the last release.
- [x] **Review and Update Tutorials**
    -   [x] Verify that the `bayesian_fitter_tutorial.rst` is accurate and works with the latest code.
    -   [x] Verify that the `seasonal_data_tutorial.rst` is accurate and works with the latest code.
    -   [x] Review all other examples and tutorials to ensure they are up-to-date.

## Model Diagnostics

- [x] **Enhance Residual Analysis**
    -   [x] Complete the implementation of robust residual analysis plots (e.g., ACF/PACF plots) in `innovate.plots.diagnostics`.

## Future Development (Post v0.3.0)

- [ ] **Phase 7: Heterogeneity & Segmentation**
    -   [ ] Implement latent-class and hierarchical models.
    -   [ ] Enhance models with covariate-driven parameterization.
- [ ] **Phase 8: Causal & Impact Assessment**
    -   [ ] Integrate with survival analysis and causal inference libraries.
- [ ] **Phase 9: Ecosystem & Domain Plugins**
    -   [ ] Add data connectors, domain-specific modules, and interactive dashboards.
