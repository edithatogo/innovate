---
title: HPC Packaging and Registry Readiness
description: Deployment and packaging evidence before HPC registry action.
slug: latest/operations/hpc-readiness
---

# HPC Packaging and Registry Readiness

The HPC readiness page defines packaging artifacts and evidence requirements for Spack, EasyBuild, HPSF, and E4S.

Readiness currently includes:

* candidate package recipes,
* scheduler evidence collection,
* install/smoke logs,
* explicit maintainer-managed handoff gates for upstream registry review.

The current state is readiness planning, not a registry claim. Spack and
EasyBuild are ready for maintainer review with scheduler evidence captured;
HPSF and E4S still require governance or accelerator-review evidence before any
submission claim.

## Install surfaces and variants

* Python package surface: build wheel and sdist, install under Python 3.14, run `python -m pip check`, and run `python -c "import innovate; print(innovate.__version__)"`.
* Rust crate and native slices: run `cargo test --manifest-path bindings/rust/Cargo.toml`; the `variant("+rust"` path requires `cargo` and `rust`.
* Optional JAX/XLA extras: the `variant("+jax"` path should include `py-jax` and `py-jaxlib`.
* Language binding surfaces: Julia, R, and TypeScript smoke checks are grouped behind `variant("+bindings"`.
* Documentation extras remain explicit with `variant("+docs"`.

Dependency floors mirror the Python 3.14 lockfile: `py-numpy@2.4.4:2`,
`py-scipy@1.17.1:1`, `py-pandas@3.0.2:3`, `py-pyarrow@23.0.1:23`,
`py-statsmodels@0.14.6:0.14`, `py-mesa@3.5.1:3`, `py-networkx@3.6.1:3`,
`py-ndlib@5.1.1:5`, `py-jitcdde@1.8.3:1`, `py-sympy@1.14:1`,
`py-ruptures@1.1.9:1.1.9`, `py-pymannkendall@1.4.3:1`, and
`py-pytensor@2.38.2:2`.

## Deployment options

* CPU-only deployment records wheel/sdist install, dependency resolution, and a minimal kernel smoke call.
* GPU/XLA deployment records XLA backend metadata and whether execution used CPU fallback or a real GPU.
* Mixed Rust/Python bridge deployment records Rust-native slices plus Python bridge fallback.

## Candidate packages

* Spack package candidate: `class PyInnovate(PythonPackage):`.
* EasyBuild easyconfig candidate: `easyblock = 'PythonPackage'`.
* Slurm and PBS job templates preserve scheduler execution shape.
* HPSF and E4S evidence templates preserve maintainer handoff notes.
* The per-target command checklist and module sanity checks include
  `julia --project=bindings/julia -e`, `Rscript -e`, and
  `npm test --prefix bindings/typescript`.

Python 3.14-only package metadata is the intended baseline. If Spack,
EasyBuild, HPSF, E4S, or a downstream center cannot yet consume Python
3.14-only packages, that limitation is an external compatibility constraint.

## Evidence gates

Evidence includes `spack-batch.log`, `easybuild-batch.log`, `spack-pbs.log`,
`easybuild-pbs.log`, `evidence/hpsf-review-note.md`, and
`evidence/e4s-review-note.md`. HPSF candidacy and E4S candidacy require install,
smoke, and batch evidence. The current state says install, smoke, and batch evidence is now present. package sketches, local evidence, and batch logs are present, and performance portability evidence still governs E4S. Execution
execution templates for Slurm and PBS scheduler submission are present for CPU, GPU, and
mixed bridge evidence. No HPSF or E4S submission should be made without the
maintainer handoff note and `ready_for_maintainer` state.
Scheduler evidence must identify Slurm or PBS execution context.
The evidence bundle distinguishes CPU, GPU, and mixed bridge deployments.
no HPSF or E4S submission should be made without recorded maintainer evidence.
