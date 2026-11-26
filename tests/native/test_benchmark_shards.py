"""Exercise Slurm receipt validation without a scheduler allocation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_slurm_benchmark_receipts_require_a_single_step(tmp_path):
    runner = ROOT / "scripts/run_native_benchmarks.py"
    verifier = ROOT / "scripts/check_benchmark_shards.py"
    serial = tmp_path / "serial.json"
    template = tmp_path / "rank-{rank}.json"

    def run(script, *args, **variables):
        return subprocess.run(
            [sys.executable, script, *args],
            env={**os.environ, **variables},
            capture_output=True,
            text=True,
            check=False,
        )

    assert run(runner, "--policy", "fast", "--output", serial).returncode == 0

    def shard(rank, step, output):
        return run(
            runner,
            "--policy",
            "fast",
            "--slurm-shard",
            "--slurm-allocation",
            "--output",
            output,
            SLURM_PROCID=str(rank),
            SLURM_STEP_NUM_TASKS="2",
            SLURM_CPUS_PER_TASK="1",
            SLURM_JOB_ID="100",
            SLURM_STEP_ID=step,
        )

    for rank in (0, 1):
        receipt = shard(rank, "0", template)
        assert receipt.returncode == 0, receipt.stderr
    first, second = tmp_path / "rank-0.json", tmp_path / "rank-1.json"
    original = first.read_bytes()
    duplicate = shard(0, "0", template)
    assert duplicate.returncode == 2, duplicate.stderr
    assert "Shard receipt already exists." in duplicate.stderr
    assert str(tmp_path) not in duplicate.stdout + duplicate.stderr
    assert first.read_bytes() == original
    result = run(verifier, serial, second, first)
    assert result.returncode == 0, result.stderr

    other = tmp_path / "other-step.json"
    assert shard(1, "1", other).returncode == 0
    result = run(verifier, serial, first, other)
    assert result.returncode == 2
    assert "same job and step" in result.stderr


@pytest.mark.parametrize(
    ("invalid_field", "marker"),
    [
        ("SLURM_JOB_ID", "diagnostic_fixture_value"),
        ("SLURM_STEP_ID", "diagnostic_fixture_value"),
        ("SLURM_JOB_ID", "4294967296"),
        ("SLURM_JOB_ID", "-2147483649"),
        ("SLURM_STEP_ID", "１２３"),
    ],
)
def test_unstructured_slurm_identifiers_do_not_enter_receipts(tmp_path, invalid_field, marker):
    variables = {"SLURM_PROCID": "0", "SLURM_STEP_NUM_TASKS": "1", "SLURM_JOB_ID": "100", "SLURM_STEP_ID": "0"}
    variables[invalid_field] = marker
    output = tmp_path / "shard.json"
    result = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/run_native_benchmarks.py",
            "--policy",
            "fast",
            "--slurm-shard",
            "--output",
            output,
        ],
        env={**os.environ, **variables},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert marker not in result.stdout + result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("job", "step"),
    [
        ("100", "0"),
        ("100", "batch"),
        ("100", "extern"),
        ("100", "interactive"),
        ("-2147483647", "-6"),
        ("4294967295", "4294967294"),
    ],
)
def test_supported_slurm_identifier_forms_remain_visible(tmp_path, job, step):
    output = tmp_path / "shard.json"
    result = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/run_native_benchmarks.py",
            "--policy",
            "fast",
            "--slurm-shard",
            "--output",
            output,
        ],
        env={
            **os.environ,
            "SLURM_PROCID": "0",
            "SLURM_STEP_NUM_TASKS": "1",
            "SLURM_JOB_ID": job,
            "SLURM_STEP_ID": step,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    shard = json.loads(output.read_text())["metadata"]["benchmark_runner"]["mpi_shard"]
    assert (shard["job_id"], shard["step_id"]) == (job, step)
