Release readiness
=================

The release-readiness gate is the maintainer-facing summary for deciding
whether the repository is only a release candidate or is ready to publish.

Run the local report with:

.. code-block:: bash

   uv run nox -s release_readiness

The command writes ``docs/source/_static/release_readiness/readiness-report.json``
and prints the same status as JSON. The report is generated from
``docs/source/_static/release_readiness_contract.json`` and fails closed when
required evidence is missing, stale, or failing.

State boundaries
----------------

``release candidate``
   The code may be feature-complete, but at least one required quality,
   packaging, security, provenance, reproducibility, compatibility, Rust,
   docs, or binding artifact is missing, stale, or failing.

``release-ready``
   Every required artifact in the mature release contract is fresh and passing.
   This is the local and CI condition required before a maintainer should cut a
   public release.

``public release``
   A maintainer-approved tag, signed or otherwise approved release artifact,
   and published release notes exist. Release-ready status is a prerequisite;
   it is not the same as publication.

``external acceptance``
   External registries, scientific venues, and HPC ecosystems have accepted or
   published the package. External acceptance is tracked separately from
   release readiness because it depends on registry credentials, review queues,
   and third-party maintainers.

Interpreting readiness-report.json
----------------------------------

The report includes:

* ``overall_status``: ``release_ready`` only when every required evidence item
  passes.
* ``release_state``: ``release_candidate`` while any required evidence blocks
  release readiness.
* ``missing_evidence``: required artifacts that were not found.
* ``stale_evidence``: artifacts older than the contract allows or explicitly
  marked stale.
* ``failing_evidence``: artifacts whose status is failing, unknown, deferred,
  manual-only, or otherwise not a passing evidence status.

Use this report as the input to release review. Do not treat a registry
submission receipt, HPC readiness dossier, or external acceptance note as a
substitute for the mature release gate.
