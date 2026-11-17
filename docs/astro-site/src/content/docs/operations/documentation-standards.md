---
title: Documentation standards and research communities
description: Current standards sources, mapped documentation and remaining research-software evidence.
---

Innovate uses research-software review guidance to make its documentation useful,
reproducible and clear about scientific limits. This page maps that guidance to
current repository evidence. It records neither community approval nor complete
conformance to every standard.

The official guidance was reviewed by an agent on **2026-11-16**.
That dated assessment is not a human author declaration or a live provider check.
Recheck official policies before any application or submission.

## Current issue commitments

[Issue #398](https://github.com/edithatogo/innovate/issues/398) owns three registry deliverables.
The following states come from the existing readiness record; its validator
checks retained evidence integrity without authenticating provider responses.

| Destination | Owning issue | Recorded state |
| --- | --- | --- |
| Software Heritage | [#399](https://github.com/edithatogo/innovate/issues/399) | `pending` |
| RRID | [#400](https://github.com/edithatogo/innovate/issues/400) | `pending` |
| JOSS | [#401](https://github.com/edithatogo/innovate/issues/401) | `pending` |

Follow the [registry readiness policy](/innovate/operations/registry-readiness/)
for selected-release identity, provider receipts, author evidence and unresolved
work. Scientific qualification and release acceptance remain separate.

[Issue #556](https://github.com/edithatogo/innovate/issues/556) and [#559](https://github.com/edithatogo/innovate/issues/559)
concern scientific ecosystem and consumer interoperability qualification.
They do not establish admission to an external review community.

## Standards mapped to documentation

Each entry below is **mapped, not qualified**. Evidence links identify material
to inspect; named check scripts are routes to run, not passing results. Python,
Julia and R retain their native documentation conventions, with Rust as the
scientific core. Stricter local rules remain Innovate policy.

### Software Heritage archival

Archive the cloneable source origin and identify the selected archived release or revision with a resolving SWHID.

**Current issue commitment.** Source: [Software Heritage archival guidance](https://www.softwareheritage.org/how-to-archive-reference-code/). Applicable section: Archive and reference code.

Mapped evidence: [docs/registry-readiness.md](https://github.com/edithatogo/innovate/blob/main/docs/registry-readiness.md); [conductor/tracks/research_software_registry_readiness_20260721/registry-evidence.md](https://github.com/edithatogo/innovate/blob/main/conductor/tracks/research_software_registry_readiness_20260721/registry-evidence.md).

Check routes: [scripts/check_registry_readiness.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_registry_readiness.py).

Still required:

- Select and qualify the release; retain the provider event and resolver evidence required by the existing readiness policy.

### RRID resource identity

Check for an existing resource, describe its name, URL and purpose, and retain curator approval before advertising an assigned RRID.

**Current issue commitment.** Source: [SciCrunch curation guidance](https://scicrunch.org/scicrunch/about/Curation%20Guidelines). Applicable section: Submission of new resources.

Mapped evidence: [docs/registry-readiness.md](https://github.com/edithatogo/innovate/blob/main/docs/registry-readiness.md); [conductor/tracks/research_software_registry_readiness_20260721/registry-evidence.md](https://github.com/edithatogo/innovate/blob/main/conductor/tracks/research_software_registry_readiness_20260721/registry-evidence.md).

Check routes: [scripts/check_registry_readiness.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_registry_readiness.py).

Still required:

- Complete the live identity and duplicate check and retain any actual curator decision and resolving identifier.

### JOSS documentation

Explain the research need, audience and alternatives; document installation, useful examples, API inputs and outputs, contribution, issue reporting and support.

**Current issue commitment.** Source: [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html). Applicable section: Documentation, functionality and tests.

Mapped evidence: [README.md](https://github.com/edithatogo/innovate/blob/main/README.md); [CONTRIBUTING.md](https://github.com/edithatogo/innovate/blob/main/CONTRIBUTING.md); [LICENSE](https://github.com/edithatogo/innovate/blob/main/LICENSE); [docs/astro-site/src/content/docs/user-guide/installation.md](https://github.com/edithatogo/innovate/blob/main/docs/astro-site/src/content/docs/user-guide/installation.md); [docs/contracts/api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/api-documentation.json); [docs/contracts/content-quality.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/content-quality.json).

Check routes: [scripts/check_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_api_documentation.py); [scripts/check_documentation_content.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_documentation_content.py).

Still required:

- Review the substantive need and alternatives for the selected software revision and qualify its installed functionality and examples.

### JOSS research and author evidence

Prepare a current software paper and evidence of actual research use, sustained public development, appropriate packaging and author responsibility.

**Current issue commitment.** Source: [JOSS submission guidance](https://joss.readthedocs.io/en/latest/submitting.html), [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html). Applicable section: Submission requirements and pre-review screening.

Mapped evidence: [CITATION.cff](https://github.com/edithatogo/innovate/blob/main/CITATION.cff); [conductor/tracks/research_software_registry_readiness_20260721/joss-readiness.md](https://github.com/edithatogo/innovate/blob/main/conductor/tracks/research_software_registry_readiness_20260721/joss-readiness.md); [docs/registry-readiness.md](https://github.com/edithatogo/innovate/blob/main/docs/registry-readiness.md).

Check routes: [scripts/check_registry_readiness.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_registry_readiness.py).

Still required:

- Retain the current manuscript, qualified release, research-use and development-history evidence, author declarations and eventual editorial outcome in the owning readiness track.
- Human authors must confirm authorship and their actual design, review and AI-use declarations; a metadata check supplies none of these attestations.

### pyOpenSci documentation guidance

Provide online installation, quickstart and API documentation with standard-format docstrings, contributor guidance, a code of conduct, a license and CI evidence.

**Adopted guidance.** Source: [pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html), [pyOpenSci policies](https://www.pyopensci.org/software-peer-review/our-process/policies.html). Applicable section: Editor checks, human oversight and maintenance.

Mapped evidence: [README.md](https://github.com/edithatogo/innovate/blob/main/README.md); [CONTRIBUTING.md](https://github.com/edithatogo/innovate/blob/main/CONTRIBUTING.md); [CODE_OF_CONDUCT.md](https://github.com/edithatogo/innovate/blob/main/CODE_OF_CONDUCT.md); [LICENSE](https://github.com/edithatogo/innovate/blob/main/LICENSE); [docs/astro-site/src/content/docs/user-guide/getting-started.mdx](https://github.com/edithatogo/innovate/blob/main/docs/astro-site/src/content/docs/user-guide/getting-started.mdx); [docs/contracts/api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/api-documentation.json).

Check routes: [scripts/check_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_api_documentation.py); [scripts/check_documentation_content.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_documentation_content.py).

Still required:

- Assess package fit, current distribution and actual installation; retain meaningful human oversight and a maintainer commitment before considering submission.
- Coordinate any future journal or pyOpenSci review with the maintainer; no submission or acceptance is recorded by this mapping.

### rOpenSci documentation guidance

Document exported functions, defaults, return types and examples using roxygen2; provide package help, useful cross-links and a substantial realistic HTML vignette.

**Adopted guidance.** Source: [rOpenSci packaging guide](https://devguide.ropensci.org/pkg_building.html). Applicable section: 1.7 Documentation.

Mapped evidence: [bindings/r/R/kernel_bridge.R](https://github.com/edithatogo/innovate/blob/main/bindings/r/R/kernel_bridge.R); [bindings/r/man/innovate.R-package.Rd](https://github.com/edithatogo/innovate/blob/main/bindings/r/man/innovate.R-package.Rd); [bindings/r/vignettes/innovate-r-kernel.Rmd](https://github.com/edithatogo/innovate/blob/main/bindings/r/vignettes/innovate-r-kernel.Rmd); [docs/contracts/polyglot-api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/polyglot-api-documentation.json).

Check routes: [scripts/check_polyglot_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_polyglot_api_documentation.py); [scripts/check_r_documentation.R](https://github.com/edithatogo/innovate/blob/main/scripts/check_r_documentation.R).

Still required:

- Qualify a substantial installed-R vignette and its examples; the presence of a starter vignette or rendered help does not establish execution or reviewer acceptance.

### Scientific assumptions and reporting

Cite primary methods and explain applicable assumptions, convergence behaviour, calendar units, missingness and network representation; record standards that do not apply.

**Adopted guidance.** Source: [rOpenSci statistical standards](https://stats-devguide.ropensci.org/standards.html). Applicable section: G1.0; RE1.4, RE3.0-RE3.3; TS1.8, TS2.0-TS2.1; NW2.3-NW2.7.

Mapped evidence: [docs/astro-site/src/content/docs/native-migration.md](https://github.com/edithatogo/innovate/blob/main/docs/astro-site/src/content/docs/native-migration.md); [docs/contracts/scientific-pedagogical-review.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/scientific-pedagogical-review.json); [migration-ledger.json](https://github.com/edithatogo/innovate/blob/main/migration-ledger.json).

Check routes: [scripts/check_scientific_review_contract.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_scientific_review_contract.py).

Still required:

- Review each selected standard against supported native behaviour. Public-data fits, synthetic recovery and a consistent review record do not establish causal identification or empirical scientific validity.

### Python documentation conventions

Use a declared readable docstring convention and document input meaning, units, defaults, result structure, useful errors and examples consistently with the source.

**Adopted guidance.** Source: [NumPy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html). Applicable section: Docstring standard and sections.

Mapped evidence: [docs/contracts/api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/api-documentation.json); [bindings/python/innovate/__init__.py](https://github.com/edithatogo/innovate/blob/main/bindings/python/innovate/__init__.py).

Check routes: [scripts/check_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_api_documentation.py); [scripts/check_native_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_native_documentation.py).

Still required:

- Review scientific meaning and native-translated failures beyond direct Python raises. NumPy treats Raises as optional; any stronger Innovate rule is a project policy, not a universal community requirement.

### Rust documentation conventions

Provide crate guidance, meaningful examples or applicable cross-links, and relevant Errors, Panics and Safety documentation using rustdoc.

**Adopted guidance.** Source: [Rust API documentation guidelines](https://rust-lang.github.io/api-guidelines/documentation.html). Applicable section: C-CRATE-DOC, C-EXAMPLE, C-FAILURE.

Mapped evidence: [docs/contracts/api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/api-documentation.json); [rust/crates/innovate-core/src/lib.rs](https://github.com/edithatogo/innovate/blob/main/rust/crates/innovate-core/src/lib.rs).

Check routes: [scripts/check_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_api_documentation.py).

Still required:

- Run the separately required pinned Rust compiler and doctest gates; the Python documentation checker cannot establish Rust compilation or scientific correctness.

### Julia documentation conventions

Attach Markdown docstrings with signatures and defaults, clear summaries, useful related-function links and reproducible examples.

**Adopted guidance.** Source: [Julia documentation manual](https://docs.julialang.org/en/v1/manual/documentation/). Applicable section: Writing documentation.

Mapped evidence: [bindings/julia/src/Innovate.jl](https://github.com/edithatogo/innovate/blob/main/bindings/julia/src/Innovate.jl); [docs/contracts/polyglot-api-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/polyglot-api-documentation.json).

Check routes: [scripts/check_polyglot_api_documentation.py](https://github.com/edithatogo/innovate/blob/main/scripts/check_polyglot_api_documentation.py); [scripts/check_julia_documentation.jl](https://github.com/edithatogo/innovate/blob/main/scripts/check_julia_documentation.jl).

Still required:

- Qualify examples in the installed Julia environment. Julia permits concise argument descriptions for simple functions; stronger binding contracts are Innovate policy.

### Community stewardship guidance

Publish contribution and conduct expectations; assess actual community participation, leadership, governance and roadmap evidence for any future affiliation or fiscal sponsorship.

**Adopted guidance.** Source: [NumFOCUS project requirements](https://numfocus.org/projects-overview). Applicable section: Requirements to join and application status.

Mapped evidence: [CONTRIBUTING.md](https://github.com/edithatogo/innovate/blob/main/CONTRIBUTING.md); [CODE_OF_CONDUCT.md](https://github.com/edithatogo/innovate/blob/main/CODE_OF_CONDUCT.md); [conductor/archive/community_submission_readiness_20260507/spec.md](https://github.com/edithatogo/innovate/blob/main/conductor/archive/community_submission_readiness_20260507/spec.md).

Still required:

- Reassess scope and community evidence before any application. The official page observed on 16 November 2026 pauses new applications; refresh that observation before acting.

## Historical community aspirations

The [conductor/archive/community_submission_readiness_20260507/spec.md](https://github.com/edithatogo/innovate/blob/main/conductor/archive/community_submission_readiness_20260507/spec.md) records earlier targets: Apache Arrow, PyPA, pyOpenSci, rOpenSci, JOSS, NumFOCUS, scikit-learn-contrib, .NET Foundation, Julia/R community.

Those plans require reassessment against the current Rust/Polars implementation.
Historical completion labels and readiness flags remain evidence of their
original scope. They do not grant current admission, registration, affiliation
or review acceptance. Removing obsolete site pages does not remove that source
history or its outstanding obligations.

## Source access and limits

The Software Heritage and SciCrunch guidance was available through indexed
official-page excerpts during this review. Direct access timed out or was
blocked; no live archive entry, duplicate-resource search or curator decision
was verified. The other linked guidance pages were read directly.

The selected rOpenSci statistical rules are an applicability map, not a claim
to satisfy the full statistical review programme. Documentation presence,
executed examples, scientific review, empirical validity and external outcomes
remain distinct. A public-data fit does not demonstrate causal identification.

## Checking and maintaining this mapping

The [docs/contracts/community-documentation.json](https://github.com/edithatogo/innovate/blob/main/docs/contracts/community-documentation.json) is the source for this page.
Use the selected Python 3.15.2 environment from the repository's contributor
instructions:

```sh
.venv-native/bin/python scripts/check_community_documentation.py --json
```

The check validates official source identities, review dates, issue ownership,
required repository references and this page's exact projection. It delegates
registry evidence to the existing readiness checker and reports source hashes.
It does not execute the listed language/example checks, contact providers,
submit an application or establish scientific or release acceptance.

After reviewing a change to the mapping or registry record, regenerate this
page with `--write-page`, then run the normal check. Regeneration updates the
projection; it supplies no new human, runtime or provider evidence.
