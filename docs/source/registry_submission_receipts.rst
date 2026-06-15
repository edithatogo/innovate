Registry Submission Receipts
============================

This page records the registry submission receipt bundle for the binding and
HPC publication targets.

Submitted targets
-----------------

The following targets have live public registry receipts:

* PyPI/TestPyPI
* npm
* crates.io
* R-universe
* Julia General submission trigger
* Go modules
* NuGet

Pending and review-ready targets
--------------------------------

The following targets remain explicitly non-submitted and are tracked in the
closure inventory:

* CRAN - deferred until maintainer send.
* Spack - ready_for_review; upstream submission remains maintainer-managed.
* EasyBuild - ready_for_review; upstream submission remains maintainer-managed.
* HPSF - ready_for_maintainer; public proposal path identified, pending TAC
  sponsors and a maintainer-opened proposal issue.
* E4S - ready_for_maintainer; public contact/contribution path identified,
  pending E4S review/CI validation and maintainer inclusion request.

CRAN has a prepared Gmail submission draft attached to the 0.5.0 source
tarball; it is still awaiting maintainer send.

The machine-readable evidence bundle is stored at
``docs/source/_static/registry_submission_receipts.json``.
The receipt and owner-backed deferral ledger is stored at
``docs/source/_static/external_acceptance_deferrals.json``.
The target-level closure inventory is stored at
``docs/source/_static/external_submission_target_inventory.json``.
Prepared handoff packets live at
``docs/source/_static/scientific_submission_packet.json`` and
``docs/source/_static/hpc_packaging/submission_packet.json``.
