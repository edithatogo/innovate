Submission Readiness Dossiers
=============================

This page collects the repository evidence that closes the remaining submission
and governance gaps surfaced in the readiness matrices. It is intentionally
evidence-driven: each section points to implemented files rather than
speculating about future work.

pyOpenSci
---------

The repository already provides the pieces a pyOpenSci reviewer would expect:

* a concise project front door in ``README.md``;
* canonical API and tutorial documentation in ``docs/source/index.rst`` and
  ``docs/source/tutorials.rst``;
* issue templates in ``.github/ISSUE_TEMPLATE/`` for bug reports, feature
  requests, model issues, and performance issues;
* continuous testing and packaging gates in ``.github/workflows/ci.yml`` and
  ``.github/workflows/bindings-publish.yml``;
* maintenance and security policy in ``CONTRIBUTING.md`` and ``SECURITY.md``.

Scope statement: ``innovate`` is a contract-first diffusion modeling library
with a thin Python surface and language bindings layered over the same kernel
contract. It is not a monolithic notebook collection or a thin wrapper around a
single estimator API.

rOpenSci
--------

The R binding now has the reviewer evidence needed for a package review
conversation:

* package metadata in ``bindings/r/DESCRIPTION`` and ``bindings/r/NAMESPACE``;
* package usage, installation, and release-check guidance in
  ``bindings/r/README.md``;
* a local check story in ``bindings/r/cran-comments.md``;
* manuals, vignettes, and tests under ``bindings/r/man/``,
  ``bindings/r/vignettes/``, and ``bindings/r/tests/``;
* CI/package publication documentation in ``docs/source/binding_publication_ci.rst``.

Validation evidence: the package builds cleanly with ``R CMD build bindings/r``
and passes ``R CMD check --as-cran --no-manual`` with a single new-submission
NOTE, which is expected for an initial submission.
The maintainer R-universe registry repo at
``edithatogo.r-universe.dev`` now includes ``innovate.R`` and will index it
asynchronously.

JOSS
----

The paper and supporting docs already contain the core JOSS submission
bundle:

* manuscript sources in ``paper.qmd``, ``documents/manuscript_v2.md``, and
  ``documents/abstract_v2.md``;
* narrative overview, methods, and results in the paper source;
* reproducible usage and API documentation in ``README.md`` and
  ``docs/source/tutorials.rst``;
* benchmark and diagnostics coverage in the docs tree;
* citation and governance metadata in ``CITATION.cff`` and the repository
  policy docs.

The paper source should be treated as the submission bundle. The remaining
editorial work is the usual JOSS packaging step, not missing repository
evidence.

Apache Arrow
------------

The Arrow-facing contract is already explicit:

* ``docs/adr/0001-array-api-and-arrow-foundation.md`` defines the long-lived
  Arrow-compatible foundation;
* ``docs/adr/0005-heoml-schema-placement.md`` keeps schema ownership aligned
  with the repo contract;
* ``docs/source/tutorials/arrow_interchange.rst`` explains the interchange
  payloads and their versioning;
* ``docs/source/abi_binary_compatibility_strategy.rst`` describes the Arrow C
  Data Interface boundary and its non-goals;
* ``tests/unit/test_arrow_interchange.py`` validates the current contract.

The remaining Arrow work is external alignment, not missing repo-side contract
definition.

.NET Foundation
---------------

The C# binding and governance posture are now documented enough for a .NET
community conversation:

* ``bindings/csharp/README.md`` explains the thin binding contract;
* ``docs/source/tutorials/csharp_bindings.rst`` explains the launcher and
  package model;
* ``docs/source/binding_publication_ci.rst`` covers publication and CI gates;
* ``docs/source/external_governance_sustainability.rst`` now includes the
  support matrix and funding statement;
* ``tests/unit/test_csharp_binding_docs.py`` checks that the docs stay aligned.

The repo still does not claim .NET Foundation affiliation. What is complete is
the repository-side evidence and support posture.

NumFOCUS
--------

The NumFOCUS-facing governance evidence is now explicit:

* ``docs/source/external_governance_sustainability.rst`` includes the support
  matrix and funding/sustainability statement;
* ``docs/source/scientific_hpc_readiness_roadmap.rst`` links the governance
  work to the broader roadmap;
* ``conductor/tracks.md`` records the repository stewardship and track history.

This completes the repo-side governance dossier. Any future outreach would be a
separate stewardship decision, not a missing documentation gap.

R community
-----------

The R community readiness evidence is already in place:

* ``bindings/r/README.md`` documents installation, use, and release checks;
* ``bindings/r/tests/`` contains the integration coverage;
* ``bindings/r/vignettes/`` and ``bindings/r/man/`` provide user-facing
  examples and documentation;
* the verified ``R CMD build`` / ``R CMD check`` run confirms the package is
  check-clean apart from the expected NOTE for a first submission.

scikit-learn-contrib
--------------------

This project is intentionally not an estimator-collection project. The scope is
diffusion and policy modeling with a kernel-first architecture and thin
bindings. That makes scikit-learn-contrib a deliberate non-target unless the
project scope changes in the future.

The scope decision is complete, so there is no remaining blocker to resolve.
