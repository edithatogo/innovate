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
The current target-level submission closure inventory is
:download:`external_submission_target_inventory.json <_static/external_submission_target_inventory.json>`.
No submission claims readiness without evidence; every target records explicit
status, evidence links, blockers, and sequencing.

Readiness is not submission. Targets marked ready in this matrix are
``ready_for_review`` in the closure inventory unless a durable external review
URL, receipt, or acceptance record exists.

Readiness summary
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 34 26

   * - Target
     - Status
     - Current reviewer evidence
     - Current gap
   * - pyOpenSci
     - Ready
     - Python package metadata, docs, tests, examples, changelog, conduct,
       contribution, and security files.
     - None.
   * - Apache Arrow
     - Ready
     - Arrow interchange docs, ADRs, schema fixtures, and tests.
     - None.
   * - rOpenSci
     - Ready
     - R package files, manuals, vignette path, NEWS, publication checks, and
       local ``R CMD check --as-cran`` evidence.
     - None.
   * - R community
     - Ready
     - R package docs, examples, vignette path, manual checks, and release
       notes.
     - None.
   * - JOSS
     - Ready
     - Scientific docs, tests, tutorials, benchmark pages, and
       ``paper.qmd``/manuscript draft sources.
     - None.
   * - Julia community
     - Ready
     - Julia project metadata, package docs, tests, tutorial, and binding
       publication notes.
     - None.
   * - .NET Foundation
     - Ready
     - C# binding, tutorial, package publication docs, support matrix, and
       governance dossier.
     - None.
   * - scikit-learn-contrib
     - Not applicable
     - Python API, tests, fitters, tutorials, and benchmark docs.
     - The project remains a diffusion-model library rather than an estimator
       collection, so contrib submission is not a target.
   * - NumFOCUS
     - Ready
     - Conduct, contribution, security, strategic roadmap signals, and the governance/funding dossier.
     - None.

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
external governance. Those items are dependencies when they affect community
submission claims.

Reviewer evidence checklist
---------------------------

Every target in the matrix records evidence for:

* docs;
* tests;
* examples;
* citation;
* governance;
* maintenance.

Targets with missing or dependent evidence are marked ``near_ready`` or
``not_applicable`` rather than ready. The matrix is intentionally conservative
so reviewers can see the difference between available evidence and remaining
submission work.
