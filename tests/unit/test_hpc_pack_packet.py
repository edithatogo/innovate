"""Tests for the HPC packet generator."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

PACK_SCRIPT = Path("docs/source/_static/hpc_packaging/pack_packet.py")


def test_hpc_pack_packet_script_exists() -> None:
    assert PACK_SCRIPT.is_file()


def test_hpc_pack_packet_manifest_shape() -> None:
    namespace = runpy.run_path(str(PACK_SCRIPT))
    manifest = namespace["build_manifest"]()

    assert manifest["schema_version"] == 1
    assert "artifacts" in manifest
    assert "targets" in manifest
    assert len(manifest["targets"]) == 4

    rendered = json.dumps(manifest)
    assert "spack/py-innovate.py" in rendered
    assert "scheduler/slurm/spack-smoke.sbatch" in rendered
    assert "governance/hpsf-evidence.md" in rendered
