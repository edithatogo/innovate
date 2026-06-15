HPC Submission Packet
=====================

This page turns the remaining HPC registry work into an execution packet.
The packet is not a submission claim; it is a list of the concrete artifacts
needed to move each target from readiness into an upstream review or registry
action.

Packet contents
---------------

.. list-table::
   :header-rows: 1
   :widths: 16 18 18 48

   * - Target
     - Status
     - Mode
     - Next action
   * - Spack
     - ready_for_review
     - candidate_recipe
     - Submit the candidate recipe upstream only after maintainer review and
       scheduler-backed evidence refresh.
   * - EasyBuild
     - ready_for_review
     - candidate_easyconfig
     - Submit the candidate easyconfig upstream only after maintainer review
       and scheduler-backed evidence refresh.
   * - HPSF
     - ready_for_maintainer
     - governance_packet
     - Identify two HPSF TAC sponsors, complete the proposal template, and
       open the HPSF TAC GitHub proposal issue.
   * - E4S
     - ready_for_maintainer
     - performance_portability_packet
     - Contact E4S, complete review/CI validation, and open an inclusion
       request only after review evidence exists.

Machine-readable packet
-----------------------

The corresponding JSON packet lives at
``docs/source/_static/hpc_packaging/submission_packet.json``.
The target-level closure inventory lives at
``docs/source/_static/external_submission_target_inventory.json``.
Each target entry records the maintainer owner, external action URL,
requirement sources, receipt rule, and revisit condition needed to move from
readiness to a submitted or accepted external state.

Execution templates
-------------------

The HPC targets now have scheduler templates and governance evidence
templates alongside the candidate package sketches:

* ``scheduler/slurm/spack-smoke.sbatch``
* ``scheduler/slurm/easybuild-smoke.sbatch``
* ``scheduler/pbs/spack-smoke.pbs``
* ``scheduler/pbs/easybuild-smoke.pbs``
* ``governance/hpsf-evidence.md``
* ``governance/e4s-evidence.md``
* ``pack_packet.py`` - JSON manifest generator for review and handoff
* ``workflow_manifest.json`` - per-target commands and artifact destinations
* ``evidence/spack-batch.log``
* ``evidence/easybuild-batch.log``
* ``evidence/spack-pbs.log``
* ``evidence/easybuild-pbs.log``
* ``evidence/hpsf-review-note.md``
* ``evidence/e4s-review-note.md``

Evidence anchors
-----------------

The packet is anchored to the current HPC evidence bundle:

* ``docs/source/_static/hpc_packaging/evidence/python-install.log``
* ``docs/source/_static/hpc_packaging/evidence/python-smoke.log``
* ``docs/source/_static/hpc_packaging/evidence/r-build.log``
* ``docs/source/_static/hpc_packaging/evidence/r-check.log``
* ``docs/source/_static/hpc_packaging/evidence/rust-test.log``
* ``docs/source/_static/hpc_packaging/evidence/julia-installed-smoke.log``
* ``docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json``
* ``docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log``

This packet exists to keep the remaining HPC work executable and auditable.
Spack and EasyBuild are review-ready but not submitted; HPSF and E4S are
ready for maintainer-managed external proposal/contact steps but not submitted.
