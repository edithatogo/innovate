---
title: 'Innovate: a native library for innovation and policy diffusion modelling'
tags:
  - Rust
  - Python
  - Julia
  - R
  - innovation diffusion
  - technology adoption
  - research software
authors:
  - name: Dylan A. Mordaunt
    orcid: 0000-0002-9775-0603
    affiliation: '1'
affiliations:
  - name: Victoria University of Wellington
    index: 1
date: 18 November 2026
bibliography: paper.bib
---

> Working draft: not author-approved or submission-ready. Author and affiliation
> metadata are provisional, copied from the repository's citation record.
> Research-use evidence, declarations, scientific qualification and release
> acceptance remain outstanding. This draft describes development version
> 0.7.0a1 at commit `ac6fb48be0aa1ab74d31ef5515f0ef77b925bb2a`.

# Summary

Innovate is a scientific software library for studying how technologies,
products and policies are adopted, compete and replace one another. Its intended
users include researchers who need to fit adoption trajectories, compare model
scenarios and examine the assumptions behind forecasts. The current development
line places scientific computation in Rust and exposes Python, Julia and R
interfaces. Polars tables provide the Python data interface, with Arrow IPC
connecting tabular inputs to the native implementation.

The repository contains bounded implementations of sixteen model families,
including Bass, Logistic, Gompertz, Fisher–Pry, competition, network and policy
diffusion models. Available operations and limitations are documented through
capability discovery and scientific conventions. This manuscript concerns a
breaking migration, not a qualified stable release: all 38 recorded migration
obligations remain required, and the release gate remains closed. Implemented
operations do not establish complete family coverage or empirical validity.

# Statement of need

Diffusion modelling connects assumptions about adoption to observable change.
The Bass model represents first purchases through innovation and imitation
mechanisms [@bass1969], while the Fisher–Pry model addresses technological
substitution [@fisher1971]. These examples illustrate why selecting a curve is
only part of specifying an analysis. A cumulative number of adopters, a share
of an installed base and repeated service use have different interpretations.
A curve's derivative, its interval change and the count of new unique adopters
cannot be treated as interchangeable outcomes.

Innovate's documented design addresses these distinctions through model
contracts, explicit parameter domains, saved initial conditions and defined
output quantities. Its target is a reusable implementation of innovation
semantics across several language communities. This target includes adoption,
competition and substitution, with application-specific populations, costs and
clinical assumptions remaining the researcher's responsibility. A scenario
difference is an output of specified model assumptions; it does not acquire a
causal interpretation solely because the software can calculate it.

# State of the field

Several established projects address overlapping parts of this workflow.
Mesa provides a Python framework for building, analysing and visualising
agent-based models [@mesa2025]. NDlib describes and simulates diffusion
processes on complex networks [@ndlib2018]. PyMC offers a general probabilistic
programming framework for specifying and fitting Bayesian models
[@pymc2023]. These are relevant alternatives for agent simulation, network
diffusion and statistical inference respectively; they also provide useful
independent reference implementations when a scientifically equivalent problem
can be defined.

The rationale for a separate Innovate library is the combination of
innovation-specific model conventions, a common native execution boundary and
language bindings that share that computation. Extending a general Python
framework would be appropriate for many applications, but would not directly
provide the repository's chosen Rust-owned execution contract. This is an
architectural choice, not evidence that other packages cannot solve the same
research questions. A final submission must confirm that this contribution is
useful in an actual research workflow. No comparative speed, accuracy,
scalability or feature-superiority result is claimed here.

# Software design

The implementation separates the scientific core, table processing and language
interfaces. Rust validates supported requests and performs numerical work;
bindings translate inputs and expose results. Unsupported operations return
explicit errors rather than invoking a host-language numerical substitute.
The purpose is to keep mathematical behaviour and validation rules in one
implementation. The cost is a native build and distribution requirement that
must be qualified for each advertised environment.

Polars DataFrames and LazyFrames are the Python table interface. Rust owns table
validation and predictions, while the current IPC exchange copies buffers and
collects LazyFrames at the execution boundary. Julia and R currently use a
versioned JSON contract through a native C ABI. Their tabular IPC adapters and
distributable native-library bundles remain outstanding. Panel, joint-draw and
cohort IPC fields retain caller-supplied units and provenance; decoding does not
repeat all encoder-side finite-value and label checks or establish scientific
semantics. Untrusted data therefore require additional validation. These
differences prevent an assumption of identical packaging or table capabilities
across the three bindings. Additional platforms, accelerators and backends require their
own evidence before support can be advertised.

The core reuses existing Rust packages for linear algebra, optimisation,
differential equations, graph operations and statistical computation. Simulation
execution is divided explicitly: Kairos owns scheduling, state and replay,
while Innovate owns innovation behaviours. The documented Voiage boundary
delegates value-of-information and economic analysis to that project through
scenario and forecast data. These boundaries avoid treating general simulation
or economic analysis as duplicate Innovate implementations. Dependency reuse
does not by itself establish the correctness of a model, adapter or selected
provider revision. Selected numerical dependencies now use vendored source
paths; the native core prohibits registry publication pending suitable
published dependencies. A source-distribution licence inventory does not
by itself qualify that release.

Scientific documentation distinguishes numerical execution from inferential
claims. For example, the documented model time grids require finite, ordered values;
model conventions distinguish levels from signed rates; and synthetic parameter
recovery checks cannot demonstrate performance on an external population.
Repository tests, examples and retained qualification records are evidence
sources to be assessed at their recorded revisions. A successful local check
does not transfer to another implementation, package or platform. Existing
scientific obligations, uncertainty requirements and release gates remain in
force throughout manuscript preparation.

# Research impact statement

The current preparation dossier does not establish a version-bound research
application of the native development line, an external adoption record, or a
publication whose results were produced by this revision. Accordingly, this
draft makes no claim of demonstrated research impact. Repository examples and
synthetic tests describe intended workflows and implementation checks; they are
not substitutes for an evidenced use case or independent empirical validation.

Before submission, this section needs a specific research-use record identifying
the question, software revision, data provenance, analysis steps and resulting
output or publication. Any comparator study must define equivalent observation
models and report failures as well as successful results. Existing empirical
and qualification workstreams retain that work; this manuscript does not
introduce a competing study or remove required capabilities to obtain a narrower
readiness claim. Public development history, installation evidence and the final
software archive must also be reconciled with the proposed submission. Until
these requirements are met, scientific and journal readiness remain pending.

# AI usage disclosure

This working draft used ChatGPT with GPT-6 Astra Pro for source searches,
bibliographic and policy checks, and manuscript drafting. The assistance included
reading public sources and the repository's documented architecture, selecting
references and preparing the accompanying bibliography and readiness notes.
This statement describes this drafting increment only. The complete history of
AI assistance across the software, documentation and earlier manuscripts has
not been established. The authors must complete the tool and version inventory,
identify the tasks assisted, correct this disclosure where needed, and provide
their own account of human review, validation and core design decisions. No
author attestation of those activities is supplied by this draft.

# Acknowledgements

Authorship, affiliations, contributor acknowledgements, funding and conflicts of
interest require confirmation before submission. The provisional affiliation
above reproduces the existing citation metadata and is not a statement that the
author has approved it for this paper. No funding source or absence of conflicts
is asserted here.

# References
