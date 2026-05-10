HPC registry contract
=====================

Purpose
-------

This document defines the contract that must be satisfied before ``innovate``
can be submitted to HPC-oriented registries or packaging channels. It is a
registry-facing contract, not a submission claim.

The contract covers the package surfaces, evidence bundle, and decision gates
that the HPC packaging work must preserve:

* the Python package surface and its reproducible install evidence;
* the Rust crate and any promoted native slices;
* optional JAX/XLA extras and accelerator-specific smoke checks;
* language binding smoke checks for Julia, R, and TypeScript;
* scheduler-backed deployment evidence for Slurm or PBS environments;
* registry-gated metadata for Spack, EasyBuild, HPSF, and E4S.

Contract terms
--------------

1. The repository may carry candidate package recipes and easyconfigs, but
   those artifacts do not imply upstream acceptance.
2. HPC registry claims require local install, smoke, and package evidence that
   is auditable in the repository or release artifacts.
3. A scheduler-backed execution trace is required before a registry claim can
   move beyond readiness planning.
4. CPU-only, GPU/XLA, and mixed Rust/Python bridge deployments must be
   distinguishable in the evidence bundle.
5. Bridge-backed behavior must remain explicit for surfaces that are not yet
   promoted to native HPC execution.
6. The public kernel contract remains the compatibility boundary; registry
   packaging must not redefine it.

Evidence bundle
---------------

The minimum evidence bundle is:

* build output for the Python wheel and sdist;
* a clean install and ``python -m pip check`` log;
* Rust test output for the binding layer;
* Julia installed-package smoke evidence;
* R build and ``R CMD check`` evidence;
* optional accelerator smoke evidence, when JAX/XLA is enabled;
* scheduler evidence from at least one HPC batch environment;
* package sketches for Spack and EasyBuild.

Target registry gates
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - Target
     - Current status
     - Submission gate
   * - Spack
     - Candidate recipe sketch present
     - Exercise the recipe in CI and at least one scheduler-backed environment.
   * - EasyBuild
     - Candidate easyconfig present
     - Verify module sanity checks and scheduler-backed install evidence.
   * - HPSF
     - Readiness evidence present
     - Add governance alignment, maintenance contacts, and scheduler traces.
   * - E4S
     - Readiness evidence present
     - Add accepted or reviewable package evidence plus accelerator diagnostics.

Non-goals
---------

This contract does not claim that any registry submission has already occurred.
It also does not replace the scientific roadmap, ABI policy, or polyglot
documentation architecture; it only states the HPC-facing evidence boundary.

