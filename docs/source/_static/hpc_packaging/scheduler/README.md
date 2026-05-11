# HPC Scheduler Submission Templates

This directory holds cluster-job templates for the blocked HPC submission
targets. They are not submission claims; they are execution scaffolding for a
real Slurm or PBS environment.

Templates:

- `slurm/spack-smoke.sbatch` - run the Spack candidate recipe and capture
  install + smoke evidence.
- `slurm/easybuild-smoke.sbatch` - run the EasyBuild candidate easyconfig and
  capture module sanity evidence.
- `pbs/spack-smoke.pbs` - PBS variant of the Spack submission job.
- `pbs/easybuild-smoke.pbs` - PBS variant of the EasyBuild submission job.

These templates expect a prebuilt source tree, the release tarball, and a
module environment with the relevant package manager installed. The job output
and scheduler metadata should be preserved alongside the existing evidence
bundle in `../evidence/`.
