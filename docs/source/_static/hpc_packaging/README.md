# HPC Packaging Artifacts

This directory contains concrete evidence and package sketches for the HPC
packaging and registry readiness dossier.

Artifacts:

- `spack/py-innovate.py` - candidate Spack recipe sketch
- `easybuild/innovate-0.5.0.eb` - candidate EasyBuild easyconfig sketch
- `scheduler/` - Slurm and PBS job templates for blocked HPC targets
- `governance/` - HPSF and E4S evidence templates
- `pack_packet.py` - consolidates the packet into a reviewable manifest
- `submission_packet.json` - machine-readable HPC submission packet
- `workflow_manifest.json` - per-target commands and artifact destinations
- `evidence/` - captured install and smoke logs for the supported surfaces

Evidence files:

- `evidence/python-install.log`
- `evidence/python-smoke.log`
- `evidence/r-build.log`
- `evidence/r-check.log`
- `evidence/rust-test.log`
- `evidence/uv-build.log`
- `evidence/julia-installed-smoke.log`

The Python install and smoke logs were refreshed from a clean virtual
environment, and the R build/check logs were refreshed from the current
package tarball. The goal is to keep the HPC readiness dossier tied to
repository artifacts instead of prose-only claims.

The scheduler templates are intentionally conservative: they capture the job
metadata and expected follow-up steps without claiming success on a cluster.
