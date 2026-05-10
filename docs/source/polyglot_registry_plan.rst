Polyglot registry plan
======================

Purpose
-------

This document records the registry plan for the polyglot ``innovate`` surface.
It separates package-manager publication, scientific-community submissions, and
HPC-oriented registry readiness so the repository can track each path
independently.

The plan is intentionally conservative: the repository contains readiness
evidence and package gates, but this page does not claim that any external
submission has already been completed.
It does not claim that any external submission has already been completed.
Use the readiness dossiers as submission checklists, not as proof of submission.

Registry layers
---------------

Package-manager registries
  These are the language-specific publication targets for the bindings and
  core package surfaces: PyPI/TestPyPI, npm, crates.io, R-universe/CRAN, Julia
  General, Go modules, and NuGet.

Scientific community submissions
  These are the reviewer-facing targets for pyOpenSci, rOpenSci, JOSS, NumFOCUS,
  Apache Arrow, Julia community, R community, and Julia General registry
  readiness.

HPC registries
  These are the packaging and registry targets for Spack, EasyBuild, HPSF,
  and E4S.

Recommended sequence
--------------------

1. Keep the package-manager publication gates aligned with the binding CI jobs.
2. Keep the scientific submission dossiers aligned with the readiness matrices.
3. Keep the HPC contract aligned with package sketches and scheduler evidence.
4. Treat external registry submission as a release decision, not a doc-only
   milestone.
   This is a release decision, not a doc-only milestone.

Current status by layer
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Layer
     - Current status
     - Next action
   * - Package-manager registries
     - Ready for gated publication
     - Run the release workflow and submit only when registry secrets and
       manual release intent are present.
   * - Scientific community submissions
     - Readiness documented
     - Use the readiness dossiers as submission checklists, not as proof of
       submission.
   * - HPC registries
     - Readiness documented
     - Use the HPC registry contract, scheduler evidence, and package sketches
       before any upstream registry claim.

Planned registry artifacts
--------------------------

The plan relies on the following repository artifacts:

* binding publication CI evidence in ``binding_publication_ci.rst``;
* community submission evidence in ``community_submission_readiness.rst``;
* HPC registry evidence in ``hpc_packaging_registry_readiness.rst``;
* governance and sustainability evidence in ``external_governance_sustainability.rst``;
* polyglot architecture guidance in ``polyglot_repo_architecture.rst``;
* the new HPC registry contract in ``hpc_registry_contract.rst``.

Non-goals
---------

This plan does not claim that any registry submission has been made, and it
does not introduce new package formats or language bindings. It only records
the path from readiness evidence to submission.

Use the HPC registry contract to keep the HPC registry path separate from
package-manager publication.
