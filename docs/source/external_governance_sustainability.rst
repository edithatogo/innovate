External governance and sustainability dossier
==============================================

Purpose
-------

This dossier records the governance, stewardship, and support evidence needed
for external scientific and foundation-style review. It is the bridge between
the current repository state and submission conversations with NumFOCUS,
.NET Foundation, pyOpenSci, rOpenSci, JOSS, and similar programs that expect
clear maintainer responsibilities and a credible continuity story.

The canonical machine-readable matrix is
:download:`external_governance_sustainability_matrix.json <_static/external_governance_sustainability_matrix.json>`.
No external submission claim should rely on implied governance; the dossier
must cite explicit files and state any remaining gaps.

Governance summary
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 34 24

   * - Governance area
     - Status
     - Current evidence
     - Remaining gap
   * - Maintainer roles
     - Ready
     - ``CODEOWNERS``, ``conductor/tracks.md``, and this dossier.
     - None.
   * - Security policy
     - Ready
     - ``SECURITY.md`` and ``CODE_OF_CONDUCT.md``.
     - None.
   * - Citation metadata
     - Ready
     - ``CITATION.cff`` and the manuscript sources.
     - None.
   * - Contributor onboarding
     - Ready
     - ``CONTRIBUTING.md``, ``CODE_OF_CONDUCT.md``, and docs navigation.
     - None.
   * - Support policy
     - Ready
     - Release policy docs, roadmap ownership pages, and the support matrix below.
     - None.
   * - Funding path
     - Ready
     - This dossier, roadmap references, and the community-maintained funding statement below.
     - None.
   * - Roadmap ownership
     - Ready
     - ``docs/source/scientific_hpc_readiness_roadmap.rst``, ``docs/architecture_modernization_roadmap.md``, and the Conductor archive.
     - None.

Maintenance responsibility matrix
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Surface
     - Primary owner
     - Notes
   * - Python core
     - Core maintainers
     - Reference API and release source of truth.
   * - R package
     - R binding maintainers
     - Publication, manual, and vignette checks remain under R-specific docs.
   * - Julia package
     - Julia binding maintainers
     - Installed-package smoke and registry metadata remain documented.
   * - Rust core and slices
     - Rust maintainers
     - Migration evidence and promotion dossiers stay in the Rust roadmap archive.
   * - TypeScript, Go, and C# bindings
     - Language binding maintainers
     - Keep package-manager metadata and smoke tests aligned with the core API.

Support matrix
--------------

The support policy is intentionally compact: each active package surface has a
named maintainer group, a publication path, and a smoke-test or check gate that
must pass before release artifacts are considered ready.

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - Surface
     - Support owner
     - Current support evidence
   * - Python core
     - Core maintainers
     - ``pyproject.toml``, the main test suite, release workflow gates, and the
       scientific roadmap docs.
   * - R package
     - R binding maintainers
     - ``bindings/r/README.md``, ``bindings/r/cran-comments.md``,
       ``bindings/r/tests/``, the PDF manual workflow, and ``R CMD check``
       evidence.
   * - Julia package
     - Julia binding maintainers
     - ``bindings/julia/README.md``, ``bindings/julia/test/``, and the
       installed-package smoke validation path.
   * - TypeScript, Go, and C# bindings
     - Language binding maintainers
     - Binding READMEs, publication CI, and language-native smoke tests.
   * - Rust core and slices
     - Rust maintainers
     - The Rust roadmap archive, profiling evidence, and bridge-contract docs.

Funding and sustainability statement
-------------------------------------

``innovate`` is currently community maintained. The repository does not claim a
dedicated sponsor or fiscal host in this dossier. If the stewardship model
changes, this statement and the supporting evidence links should be updated
before any external affiliation claim is made.

Submission readiness sequence
-----------------------------

The safest external sequence is:

1. Finalize the support matrix and funding statement.
2. Reuse the community submission dossiers for pyOpenSci, rOpenSci, JOSS, and
   NumFOCUS.
3. Link the language-specific maintenance policies to their package publication
   guides.
4. Treat foundation outreach as governance-led, not marketing-led.

This dossier intentionally avoids claiming a foundation affiliation. It exists
to make the repository's stewardship, support, and maintenance posture
discoverable before any external review.
