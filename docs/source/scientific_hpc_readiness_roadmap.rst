Scientific and HPC readiness roadmap
====================================

Purpose
-------

This roadmap converts the next maturity layer for ``innovate`` into concrete,
Conductor-managed work. The current repository has strong packaging,
multi-language CI, Arrow-oriented interchange, optional JAX/XLA acceleration,
and a Rust-core trajectory. The remaining gap is not ordinary package wiring;
it is evidence, governance, ABI discipline, HPC deployment, and community
submission readiness.

The goal is to make the project credible for scientific community review and
HPC packaging without breaking the existing public API.

Current state
-------------

.. code-block:: mermaid

   flowchart LR
     Python[Python public API] --> Kernel[Functional kernel contract]
     Kernel --> Arrow[Arrow-compatible schemas]
     Kernel --> JAX[JAX/XLA optional backend]
     Kernel --> Rust[Rust native slices]
     Kernel --> Bridge[Python bridge fallback]
     Arrow --> R[R binding]
     Arrow --> Julia[Julia binding]
     Arrow --> TS[TypeScript binding]
     Arrow --> Go[Go binding]
     Arrow --> CS[C# binding]
     Rust --> Profiling[Criterion, flamegraph, DHAT evidence]
     JAX --> GPU[GPU evidence path]
     JAX --> TPU[TPU-eligible path]

Future target state
-------------------

.. code-block:: mermaid

   flowchart LR
     API[Stable public API] --> Contract[Versioned kernel and ABI policy]
     Contract --> ArrowC[Arrow C Data Interface boundary]
     Contract --> RustCore[Rust-owned promoted core slices]
     Contract --> XLA[XLA-backed accelerator slices]
     Contract --> Dist[Distributed and scheduler-aware execution]
     RustCore --> Spack[Spack package]
     RustCore --> EasyBuild[EasyBuild easyconfig]
     XLA --> GPU[GPU runner evidence]
     XLA --> TPU[TPU runner evidence]
     Dist --> HPC[HPC deployment dossier]
     Contract --> Reviews[pyOpenSci, rOpenSci, JOSS, NumFOCUS dossiers]
     Contract --> Communities[Apache Arrow, PyPA, .NET, Julia/R communities]

Readiness matrix
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 28 22

   * - Target
     - Current signal
     - Remaining gap
     - Owning track
   * - Apache Arrow community
     - Arrow-compatible schemas and interchange are documented.
     - Produce a clearer Arrow conformance and extension-boundary dossier.
     - Community Submission Readiness Matrix
   * - PyPA
     - Modern Python packaging, ``uv``, ``nox``, CI, and version guards exist.
     - Align contributor docs with PyPA and pyOpenSci reviewer expectations.
     - Community Submission Readiness Matrix
   * - pyOpenSci
     - Python package quality gates and docs are present.
     - Prepare scope, examples, maintenance, tests, and review issue evidence.
     - Community Submission Readiness Matrix
   * - rOpenSci
     - R package, PDF manual, vignette, and publication checks exist.
     - Prepare an R reviewer dossier and statistical-software standards map.
     - Community Submission Readiness Matrix
   * - JOSS
     - The project has docs, tests, and scientific scope.
     - Add paper metadata, citations, statement of need, and comparison notes.
     - Community Submission Readiness Matrix
   * - NumFOCUS
     - Governance and multi-language scientific scope are emerging.
     - Add sustainability, governance, funding, and adoption evidence.
     - External Governance and Sustainability Dossier
   * - HPSF and E4S
     - Optional XLA and Rust profiling exist.
     - Add HPC deployment, performance portability, and registry evidence.
     - HPC Packaging and Registry Readiness
   * - Spack
     - Package versions are synchronized.
     - Add package recipe, dependency variants, and install smoke tests.
     - HPC Packaging and Registry Readiness
   * - EasyBuild
     - Package artifacts exist.
     - Add easyconfig, module sanity checks, and dependency notes.
     - HPC Packaging and Registry Readiness
   * - scikit-learn-contrib
     - Python API and test practices are mature.
     - Confirm scope fit, estimator conventions, examples, and naming policy.
     - Community Submission Readiness Matrix
   * - .NET Foundation
     - C# binding and NuGet path exist.
     - Add .NET governance, API docs, package evidence, and support policy.
     - Community Submission Readiness Matrix
   * - Julia and R communities
     - Julia and R bindings have CI and publication-readiness gates.
     - Add community-facing examples, registry notes, and maintenance owners.
     - Community Submission Readiness Matrix

SOTA and HPC gaps
-----------------

The next SOTA gaps are evidence and deployment gaps:

* accelerator evidence for CPU, GPU, TPU, and accelerator-specific backends
  such as ASIC-oriented runtimes where practical;
* distributed execution guidance for batch workloads, not a second public API;
* scheduler-aware examples for Slurm/PBS-style environments;
* reproducible Spack and EasyBuild packaging;
* native Rust promotion plans for every remaining Python-backed model slice;
* reviewer-facing community dossiers for scientific and language ecosystems;
* ABI and binary-compatibility policy for native components;
* documentation organized for users, binding authors, HPC administrators, and
  maintainers.

ABI and API compatibility
-------------------------

ABI strategy is relevant once the Rust core and HPC packaging become first
class. The public Python, R, Julia, TypeScript, Go, C#, and Rust APIs should
remain stable while native implementation details evolve behind the functional
kernel contract.

The safe boundary is:

* public APIs stay semantic-versioned and schema-versioned;
* Arrow-compatible payloads remain the durable interchange format;
* Arrow C Data Interface compatibility can be used for native interchange
  where process or language boundaries require ABI discipline;
* XLA, ``jaxlib``, Rust internal structs, and scheduler-specific details do not
  become public ABI;
* native capability discovery decides whether Rust, XLA, bridge fallback, or
  distributed execution handles a request.

XLA, ``jaxlib``, Rust internal structs, and scheduler-specific details do not become public ABI.

Follow-on tracks
----------------

The readiness roadmap is decomposed into these Conductor tracks:

* Community Submission Readiness Matrix
* HPC Packaging and Registry Readiness
* Accelerator and Parallel Execution Evidence
* Rust Core Migration Execution Plan
* ABI and Binary Compatibility Strategy
* Polyglot Repository and Documentation Architecture
* External Governance and Sustainability Dossier

Dependency graph
----------------

.. code-block:: mermaid

   flowchart TD
     A[Scientific and HPC readiness roadmap] --> B[Community submission readiness]
     A --> C[HPC packaging and registry readiness]
     A --> D[Accelerator and parallel execution evidence]
     A --> E[Rust core migration execution]
     A --> F[ABI and binary compatibility strategy]
     A --> G[Polyglot docs and repo architecture]
     A --> H[External governance and sustainability]
     F --> C
     F --> E
     D --> C
     E --> C
     G --> B
     H --> B
     C --> I[HPSF and E4S candidacy]
     B --> J[pyOpenSci, rOpenSci, JOSS, NumFOCUS submissions]

Parallel work plan
------------------

The follow-on tracks are designed for six subagents without overlapping write
ownership:

* Agent A: community submission dossiers and reviewer matrices.
* Agent B: HPC packaging, Spack, EasyBuild, HPSF, and E4S readiness.
* Agent C: accelerator and distributed execution evidence.
* Agent D: Rust core migration execution plan and promotion backlog.
* Agent E: ABI policy and polyglot documentation structure.
* Agent F: external governance, sustainability, and final cross-track review.

The dependency order is explicit: ABI policy, accelerator evidence, Rust
migration evidence, and documentation architecture feed community and HPC
submission readiness. Work can begin in parallel, but submission claims should
not be made until their dependent evidence tracks are complete.
