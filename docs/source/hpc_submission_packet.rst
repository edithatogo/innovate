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
     - blocked
     - candidate_recipe
     - Run the recipe in a scheduler-backed environment and capture the batch
       log.
   * - EasyBuild
     - blocked
     - candidate_easyconfig
     - Run the easyconfig in a scheduler-backed environment and capture the
       module sanity log.
   * - HPSF
     - blocked
     - governance_packet
     - Add governance contacts and scheduler-backed deployment evidence.
   * - E4S
     - blocked
     - performance_portability_packet
     - Add accelerator-aware smoke evidence and a reviewable package artifact
       set.

Machine-readable packet
-----------------------

The corresponding JSON packet lives at
``docs/source/_static/hpc_packaging/submission_packet.json``.

Execution templates
-------------------

The blocked HPC targets now have scheduler templates and governance evidence
templates alongside the candidate package sketches:

* ``scheduler/slurm/spack-smoke.sbatch``
* ``scheduler/slurm/easybuild-smoke.sbatch``
* ``scheduler/pbs/spack-smoke.pbs``
* ``scheduler/pbs/easybuild-smoke.pbs``
* ``governance/hpsf-evidence.md``
* ``governance/e4s-evidence.md``
* ``pack_packet.py`` - JSON manifest generator for review and handoff
* ``workflow_manifest.json`` - per-target commands and artifact destinations

Evidence anchors
-----------------

The packet is anchored to the current HPC evidence bundle:

* ``docs/source/_static/hpc_packaging/evidence/python-install.log``
* ``docs/source/_static/hpc_packaging/evidence/python-smoke.log``
* ``docs/source/_static/hpc_packaging/evidence/r-build.log``
* ``docs/source/_static/hpc_packaging/evidence/r-check.log``
* ``docs/source/_static/hpc_packaging/evidence/rust-test.log``
* ``docs/source/_static/hpc_packaging/evidence/julia-installed-smoke.log``

This packet exists to keep the remaining HPC work executable and auditable
while upstream registry claims are still blocked.
