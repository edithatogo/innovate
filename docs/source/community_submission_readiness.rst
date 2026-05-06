Community submission readiness
==============================

Purpose
-------

This dossier translates the scientific and HPC readiness roadmap into
reviewer-facing submission evidence for language and scientific communities.
It covers pyOpenSci, rOpenSci, JOSS, NumFOCUS, scikit-learn-contrib, Apache
Arrow, .NET Foundation, Julia community, and R community expectations.

The canonical machine-readable matrix is
:download:`community_submission_readiness_matrix.json <_static/community_submission_readiness_matrix.json>`.
No submission claims readiness without evidence; every target records explicit
status, evidence links, blockers, and sequencing.

Readiness summary
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 34 26

   * - Target
     - Status
     - Current reviewer evidence
     - Submission blocker
   * - pyOpenSci
     - Near-ready
     - Python package metadata, docs, tests, examples, changelog, conduct,
       contribution, and security files.
     - Needs a pyOpenSci-specific scope and review checklist.
   * - Apache Arrow
     - Near-ready
     - Arrow interchange docs, ADRs, schema fixtures, and tests.
     - Needs an Arrow-focused dossier and ABI boundary coordination before
       low-level interface claims.
   * - rOpenSci
     - Blocked
     - R package files, manuals, vignette path, NEWS, and package publication
       checks.
     - Needs a rOpenSci standards map and clean R check evidence.
   * - R community
     - Near-ready
     - R package docs, examples, vignette path, manual checks, and release
       notes.
     - Needs final R community examples and package-check evidence.
   * - JOSS
     - Blocked
     - Scientific docs, tests, tutorials, benchmark pages, and manuscript
       drafts.
     - Needs paper metadata, references, authorship, statement of need, and
       reproducibility links.
   * - Julia community
     - Near-ready
     - Julia project metadata, package docs, tests, tutorial, and binding
       publication notes.
     - Needs registration notes and installed-package bridge assumptions.
   * - .NET Foundation
     - Blocked
     - C# binding, tutorial, package publication docs, and previous package
       publication track.
     - Needs support policy, API documentation checklist, ownership model, and
       governance evidence.
   * - scikit-learn-contrib
     - Blocked
     - Python API, tests, fitters, tutorials, and benchmark docs.
     - Needs a scope-fit decision and estimator-convention dossier.
   * - NumFOCUS
     - Blocked
     - Conduct, contribution, security, and strategic roadmap signals.
     - Needs the external governance and sustainability dossier before any
       foundation-readiness claim.

Submission sequencing
---------------------

The safest sequence is:

1. pyOpenSci and Apache Arrow community conversations after their focused
   reviewer checklists are added.
2. R community and rOpenSci after R package check evidence and standards
   mapping are current.
3. JOSS after paper metadata, citation, statement of need, and reproducible
   examples are finalized.
4. Julia community after installed-package bridge expectations and registration
   notes are stable.
5. .NET Foundation and NumFOCUS after the external governance and
   sustainability dossier is complete.
6. scikit-learn-contrib only after the project explicitly decides whether the
   estimator ecosystem is a good scope fit.

Cross-track boundaries
----------------------

This dossier does not claim ownership of HPC packaging, accelerator evidence,
Rust migration execution, ABI policy, polyglot documentation architecture, or
external governance. Those items are blockers or dependencies when they affect
community submission claims.

Reviewer evidence checklist
---------------------------

Every target in the matrix records evidence for:

* docs;
* tests;
* examples;
* citation;
* governance;
* maintenance.

Targets with missing or dependent evidence are marked ``blocked`` or
``near_ready`` rather than ready. The matrix is intentionally conservative so
reviewers can see the difference between available evidence and remaining
submission work.
