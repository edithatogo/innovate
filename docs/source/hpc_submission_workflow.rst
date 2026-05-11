HPC Submission Workflow
=======================

This page turns the HPC submission arrangement into a per-target execution
checklist. It does not claim upstream submission success; it only records the
commands and artifact destinations needed to complete the handoff.

Manifest
--------

The machine-readable manifest lives at
``docs/source/_static/hpc_packaging/workflow_manifest.json``.

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

EasyBuild
~~~~~~~~~

1. Set ``INNOVATE_SDIST`` to the source tarball.
2. Set ``EASYCONFIG_PATH`` to
   ``docs/source/_static/hpc_packaging/easybuild/innovate-0.5.0.eb``.
3. Submit the Slurm job with
   ``sbatch docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch``.
4. Save the resulting install, sanity, and batch logs under
   ``docs/source/_static/hpc_packaging/evidence/``.

HPSF
~~~~

1. Populate ``docs/source/_static/hpc_packaging/governance/hpsf-evidence.md``.
2. Attach ``r-build.log`` and ``r-check.log``.
3. Record the review contact or blocker note in the evidence bundle.

E4S
~~~

1. Populate ``docs/source/_static/hpc_packaging/governance/e4s-evidence.md``.
2. Attach ``rust-test.log`` and ``julia-installed-smoke.log``.
3. Record the review contact or blocker note in the evidence bundle.

Status
------

The workflow is prepared, not submitted. The commands above define the exact
arrangement needed for the external handoff.
