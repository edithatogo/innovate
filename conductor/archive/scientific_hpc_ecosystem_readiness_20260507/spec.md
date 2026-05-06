# Specification

## Overview

Create a SOTA scientific and HPC readiness roadmap that converts community
submission, HPC deployment, accelerator evidence, ABI, Rust migration, and
polyglot documentation gaps into Conductor-managed follow-on tracks.

## Functional Requirements

1. Document current and future architecture with Mermaid diagrams.
2. Add a readiness matrix for Apache Arrow, PyPA, pyOpenSci, rOpenSci, JOSS,
   NumFOCUS, HPSF, E4S, Spack, EasyBuild, scikit-learn-contrib, .NET
   Foundation, and Julia/R communities.
3. Identify SOTA and HPC gaps, including GPU, TPU, ASIC-oriented accelerator,
   distributed execution, and scheduler-aware deployment gaps.
4. Define an API-preserving ABI strategy for native and Arrow-backed
   components.
5. Register concrete follow-on Conductor tracks with dependencies and
   subagent-ready work ownership.

## Non-Functional Requirements

1. Do not claim the library is already HPC-registry ready.
2. Keep the public API stable in the roadmap.
3. Keep the work decomposed for parallel execution.

## Acceptance Criteria

1. A Sphinx roadmap page exists and is linked from the docs index.
2. The roadmap includes current and future Mermaid diagrams.
3. The roadmap names all requested community and HPC targets.
4. Follow-on tracks exist for submission readiness, HPC packaging,
   accelerator evidence, Rust migration, ABI, polyglot docs, and governance.
5. Tests guard the roadmap and follow-on track registry.

## Out of Scope

1. Completing actual submissions to external organizations.
2. Implementing every Rust-native kernel slice.
3. Publishing Spack or EasyBuild recipes upstream.
