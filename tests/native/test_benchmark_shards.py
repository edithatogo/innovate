"""Exercise Slurm receipt validation without a scheduler allocation."""

import os
import subprocess
import sys
from pathlib import Path

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
    result = run(verifier, serial, second, first)
    assert result.returncode == 0, result.stderr

    other = tmp_path / "other-step.json"
    assert shard(1, "1", other).returncode == 0
    result = run(verifier, serial, first, other)
    assert result.returncode == 2
    assert "same job and step" in result.stderr
