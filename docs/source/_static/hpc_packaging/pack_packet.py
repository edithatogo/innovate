"""Generate a consolidated HPC submission packet manifest.

This script is intentionally conservative: it does not attempt to submit to
any registry or execute on a scheduler. It only assembles the package sketches,
scheduler templates, governance templates, and evidence anchors into a single
machine-readable summary that can be copied into a cluster job workspace or
used in release review.
"""

from __future__ import annotations

import json


def build_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": {
            "spack_recipe": "spack/py-innovate.py",
            "easybuild_easyconfig": "easybuild/innovate-0.5.0.eb",
            "scheduler_templates": [
                "scheduler/slurm/spack-smoke.sbatch",
                "scheduler/slurm/easybuild-smoke.sbatch",
                "scheduler/pbs/spack-smoke.pbs",
                "scheduler/pbs/easybuild-smoke.pbs",
            ],
            "governance_templates": [
                "governance/hpsf-evidence.md",
                "governance/e4s-evidence.md",
            ],
            "evidence": [
                "evidence/python-install.log",
                "evidence/python-smoke.log",
                "evidence/r-build.log",
                "evidence/r-check.log",
                "evidence/rust-test.log",
                "evidence/uv-build.log",
                "evidence/julia-installed-smoke.log",
            ],
        },
        "targets": [
            {
                "target_id": "spack",
                "registry": "Spack",
                "status": "blocked",
                "next_action": "Submit the candidate recipe in a scheduler-backed environment and save the batch log.",
            },
            {
                "target_id": "easybuild",
                "registry": "EasyBuild",
                "status": "blocked",
                "next_action": "Run the easyconfig in a scheduler-backed environment and save the module sanity log.",
            },
            {
                "target_id": "hpsf",
                "registry": "HPSF",
                "status": "blocked",
                "next_action": "Populate governance contacts and preserve scheduler-backed deployment evidence.",
            },
            {
                "target_id": "e4s",
                "registry": "E4S",
                "status": "blocked",
                "next_action": "Capture accelerator-aware smoke evidence and a reviewable package artifact set.",
            },
        ],
        "rendered_sources": {
            "packet": "submission_packet.json",
            "readme": "README.md",
        },
    }


def main() -> None:
    manifest = build_manifest()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
