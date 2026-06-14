HPC Submission Workflow
=======================

This page turns the HPC submission arrangement into a per-target execution
checklist. It does not claim upstream submission success; it only records the
commands and artifact destinations needed to complete the handoff.

Manifest
--------

The machine-readable manifest lives at
``docs/source/_static/hpc_packaging/workflow_manifest.json``.

Blockers
--------

The current local probe is recorded in
``docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log``.
Any remaining blockers live in
``docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json``.

Targets
-------

Spack
~~~~~

1. Set ``INNOVATE_SDIST`` to the source tarball.
2. Set ``SPACK_PKG_PATH`` to
   ``docs/source/_static/hpc_packaging/spack/py-innovate.py``.
3. Submit the Slurm job with
   ``sbatch docs/source/_static/hpc_packaging/scheduler/slurm/spack-smoke.sbatch``.
4. Save the resulting install, smoke, and batch logs under
   ``docs/source/_static/hpc_packaging/evidence/``.
   The currently captured logs are ``spack-batch.log`` and
   ``spack-pbs.log``.

EasyBuild
~~~~~~~~~

1. Set ``INNOVATE_SDIST`` to the source tarball.
2. Set ``EASYCONFIG_PATH`` to
   ``docs/source/_static/hpc_packaging/easybuild/innovate-0.5.0.eb``.
3. Submit the Slurm job with
   ``sbatch docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch``.
4. Save the resulting install, sanity, and batch logs under
   ``docs/source/_static/hpc_packaging/evidence/``.
   The currently captured logs are ``easybuild-batch.log`` and
   ``easybuild-pbs.log``.

HPSF
~~~~

1. Populate ``docs/source/_static/hpc_packaging/governance/hpsf-evidence.md``.
2. Attach ``r-build.log`` and ``r-check.log``.
3. Record the review contact or blocker note in the evidence bundle
   (currently ``evidence/hpsf-review-note.md``).

E4S
~~~

1. Populate ``docs/source/_static/hpc_packaging/governance/e4s-evidence.md``.
2. Attach ``rust-test.log`` and ``julia-installed-smoke.log``.
3. Record the review contact or blocker note in the evidence bundle
   (currently ``evidence/e4s-review-note.md``).

Status
------

The workflow is partially executable in this local environment. Spack and
EasyBuild can be exercised through the container-backed scheduler wrappers,
and their captured batch logs are preserved in the evidence bundle; they are
``ready_for_review`` rather than submitted. HPSF and E4S remain blocked on
governance and accelerator-review channels. The commands above define the
exact arrangement needed for the external handoff, and the explicit probe
output is preserved alongside the remaining blocker bundle and closure
inventory.
