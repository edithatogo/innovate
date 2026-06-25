---
title: HPC Registry Contract
description: Contract and constraints for HPC registry submission.
---

# HPC Registry Contract

This page documents the policy and evidence requirements for Spack, EasyBuild, HPSF, and E4S submission paths.

Key contract points:

- this is a registry-facing contract, not a submission claim,
- package sketches and reproducible install evidence are required,
- scheduler-backed execution evidence is required before downstream registry acceptance,
- a scheduler-backed execution trace is required before readiness moves beyond planning,
- public claims cannot exceed what the evidence set supports,
- the public kernel contract remains the compatibility boundary.

## Evidence bundle

- Python wheel and sdist build output.
- Clean install and `python -m pip check` log.
- Rust test output.
- Julia installed-package smoke evidence.
- R build and ``R CMD check`` evidence.
- Optional accelerator smoke evidence when JAX/XLA is enabled.
- Scheduler evidence from at least one HPC batch environment.
- Package sketches for Spack and EasyBuild.

Target gates cover Spack, EasyBuild, HPSF, and E4S. Spack and EasyBuild require
candidate recipe or easyconfig review plus scheduler-backed traces. HPSF and
E4S require governance, maintenance, review, and accelerator evidence before
any submission claim.
