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
must cite explicit files and state any remaining blockers.

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
     - Near-ready
     - Release policy docs and roadmap ownership pages.
     - Add a compact maintainer support matrix for active package surfaces.
   * - Funding path
     - Blocked
     - Roadmap and manuscript references.
     - Add a funding or sponsorship statement before any external affiliation claim.
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
