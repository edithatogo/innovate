# HPC Packaging Artifacts

This directory contains concrete evidence and package sketches for the HPC
packaging and registry readiness dossier.

Artifacts:

- `spack/py-innovate.py` - candidate Spack recipe sketch
- `easybuild/innovate-0.5.0.eb` - candidate EasyBuild easyconfig sketch
- `scheduler/` - Slurm and PBS job templates for HPC targets that can now be executed locally
- `governance/` - HPSF and E4S evidence templates
- `pack_packet.py` - consolidates the packet into a reviewable manifest
- `submission_packet.json` - machine-readable HPC submission packet
- `workflow_manifest.json` - per-target commands and artifact destinations
- `evidence/hpc_submission_blockers.json` - compatibility-named blocker and
  resolved-handoff status bundle
- `evidence/hpc_submission_environment_probe.log` - explicit local tool probe and wrapper state
- `evidence/` - captured install and smoke logs for the supported surfaces

Evidence files:

- `evidence/python-install.log`
- `evidence/python-smoke.log`
- `evidence/r-build.log`
- `evidence/r-check.log`
- `evidence/rust-test.log`
- `evidence/uv-build.log`
- `evidence/julia-installed-smoke.log`
- `evidence/spack-batch.log`
- `evidence/easybuild-batch.log`
- `evidence/spack-pbs.log`
- `evidence/easybuild-pbs.log`
- `evidence/hpsf-review-note.md`
- `evidence/e4s-review-note.md`

The Python install and smoke logs were refreshed from a clean virtual
environment, the R build/check logs were refreshed from the current package
tarball, and the current batch logs were captured through the local
container-backed scheduler wrappers. The goal is to keep the HPC readiness
dossier tied to repository artifacts instead of prose-only claims.

The scheduler templates are intentionally conservative: they capture the job
metadata and expected follow-up steps without claiming upstream submission
success. The environment probe log records the current local `sbatch`,
`spack`, `eb`, and `qsub` availability, including the container-backed
wrappers used for the scheduler commands.
