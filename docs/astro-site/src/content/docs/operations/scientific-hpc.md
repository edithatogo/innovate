---
title: Scientific and HPC Readiness Roadmap
description: Evidence and gates for scientific and HPC packaging maturity.
---

# Scientific and HPC Readiness Roadmap

This roadmap keeps the public API boundary stable while it advances deployment maturity for HPC and scientific review.

Current focus areas:

- Rust slice completion and capability metadata consistency.
- Native and optional JAX/XLA acceleration gates.
- Package-manager and HPC registry readiness.
- Evidence-first proof for scheduler and portability claims.

Repository evidence is treated as a gated readiness set, not an upstream acceptance claim by itself.

## Submission targets

The roadmap covers Apache Arrow, PyPA, pyOpenSci, rOpenSci, JOSS, NumFOCUS,
HPSF, E4S, Spack, EasyBuild, scikit-learn-contrib, .NET Foundation, and Julia and R communities.

## SOTA and HPC gaps

Evidence gaps cover CPU, GPU, TPU, ASIC-oriented runtimes, distributed
execution, scheduler-aware examples, Slurm/PBS-style environments, Spack and
EasyBuild packaging, native Rust promotion, community dossiers, ABI policy, and
documentation organized for users, binding authors, HPC administrators, and
maintainers.

## ABI and API compatibility

public APIs stay semantic-versioned and schema-versioned. Arrow-compatible
payloads remain the durable interchange format. The Arrow C Data Interface can
be used where native interchange needs ABI discipline. XLA, `jaxlib`, Rust
internal structs, and scheduler-specific details do not become public ABI.
XLA, ``jaxlib``, Rust internal structs, and scheduler-specific details do not become public ABI.
Native capability discovery decides whether Rust, XLA, explicit bridge
fallback, or distributed execution handles a request.

## Parallel work plan

Follow-on tracks:

- Community Submission Readiness Matrix
- HPC Packaging and Registry Readiness
- Accelerator and Parallel Execution Evidence
- Rust Core Migration Execution Plan
- ABI and Binary Compatibility Strategy
- Polyglot Repository and Documentation Architecture
- External Governance and Sustainability Dossier

Agent A owns community submission dossiers and reviewer matrices. Agent B owns
HPC packaging, Spack, EasyBuild, HPSF, and E4S readiness. Agent C owns
accelerator and distributed execution evidence. Agent D owns Rust core
migration execution. Agent E owns ABI policy and polyglot documentation
structure. Agent F owns external governance, sustainability, and final
cross-track review. Dependency graph: ABI policy, accelerator
evidence, Rust migration evidence, documentation architecture, community
readiness, and HPC candidacy.
