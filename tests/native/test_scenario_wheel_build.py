"""Guard evidence acquisition against false successful build records."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "conductor/tracks/scientific_ecosystem_alignment_20261029/integration/build_scenario_wheel.py"
spec = importlib.util.spec_from_file_location("scenario_wheel_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[build-system]\nrequires = ["maturin==1.16.0"]\n')
    core = repo / "rust/crates/innovate-core"
    core.mkdir(parents=True)
    (core / "Cargo.toml").write_text(
        '[dependencies]\nkairo-ecs-core = { git = "https://github.com/edithatogo/kairos.git", '
        'rev = "a1f6283575ad26918230adc122a99b5c08fc4da3" }\n'
    )
    (core / "source.rs").write_text("pub fn value() -> u32 { 7 }\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Evidence test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(builder.importlib.metadata, "version", lambda name: "1.16.0")
    monkeypatch.setattr(builder, "tool_version", lambda name: f"{name} 1.100.0")
    return repo


def successful_build(command, repository, log, environment):
    Path(command[-1]).mkdir()
    (Path(command[-1]) / "innovate-0.7.0a1.whl").write_bytes(b"test artifact identity")
    log.write("completed test builder\n")
    return 0


def test_record_binds_actual_command_source_artifact_and_log(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "run_build", successful_build)
    out = tmp_path / "evidence"
    record = builder.build(checkout, out, 2)
    assert record["status"] == "built"
    assert record["build_exit_code"] == 0
    assert record["build_jobs"] == 2
    assert record["wheel_sha256"] == builder.qualifier.sha256(out / "wheels/innovate-0.7.0a1.whl")
    assert record["build_log_sha256"] == builder.qualifier.sha256(out / "build.log")
    assert record["source_commit"] == builder.git(checkout, "rev-parse", "HEAD").strip()
    assert record["source_inputs"] == builder.qualifier.build_inputs(checkout)
    assert record["source_inputs_unchanged_after_build"] is True
    assert record["command_argv"][1:5] == ["-m", "maturin", "build", "--locked"]
    assert json.loads((out / "build-provenance.json").read_text()) == record


def test_failed_build_retains_log_but_never_success_record(checkout, tmp_path, monkeypatch):
    def fail(command, repository, log, environment):
        log.write("compiler rejected input\n")
        return 17

    monkeypatch.setattr(builder, "run_build", fail)
    out = tmp_path / "failure"
    with pytest.raises(ValueError, match="exit 17"):
        builder.build(checkout, out, 1)
    assert "compiler rejected" in (out / "build.log").read_text()
    assert not (out / "build-provenance.json").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_reject_missing_or_ambiguous_wheels(checkout, tmp_path, monkeypatch, count):
    def ambiguous(command, repository, log, environment):
        wheels = Path(command[-1])
        wheels.mkdir()
        for index in range(count):
            (wheels / f"innovate-{index}.whl").write_bytes(b"wheel")
        return 0

    monkeypatch.setattr(builder, "run_build", ambiguous)
    with pytest.raises(ValueError, match="exactly one"):
        builder.build(checkout, tmp_path / "evidence", 1)


def test_reject_changed_inputs_during_build(checkout, tmp_path, monkeypatch):
    def drift(command, repository, log, environment):
        code = successful_build(command, repository, log, environment)
        (repository / "pyproject.toml").write_text("changed during build")
        return code

    monkeypatch.setattr(builder, "run_build", drift)
    out = tmp_path / "evidence"
    with pytest.raises(ValueError, match="changed during build"):
        builder.build(checkout, out, 1)
    assert not (out / "build-provenance.json").exists()


@pytest.mark.parametrize("path", ["pyproject.toml", "rust/new.rs", "bindings/python/new.py"])
def test_reject_dirty_or_untracked_build_inputs(checkout, tmp_path, path):
    target = checkout / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("uncommitted build input")
    with pytest.raises(ValueError, match="uncommitted build inputs"):
        builder.build(checkout, tmp_path / "evidence", 1)


def test_documentation_edits_are_recorded_without_blocking_build(checkout, tmp_path, monkeypatch):
    (checkout / "note.md").write_text("pending evidence")
    monkeypatch.setattr(builder, "run_build", successful_build)
    record = builder.build(checkout, tmp_path / "evidence", 1)
    assert "?? note.md" in record["worktree_changes"]


def test_does_not_reuse_output(checkout, tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "keep.txt").write_text("retained")
    with pytest.raises(FileExistsError):
        builder.build(checkout, out, 1)
    assert (out / "keep.txt").read_text() == "retained"


@pytest.mark.parametrize("jobs", [0, -1, True])
def test_reject_invalid_job_count(checkout, tmp_path, jobs):
    with pytest.raises(ValueError, match="positive integer"):
        builder.build(checkout, tmp_path / "evidence", jobs)


@pytest.mark.parametrize("python_version,maturin", [((3, 15, 1), "1.16.0"), ((3, 15, 2), "1.15.0")])
def test_reject_wrong_build_runtime(checkout, python_version, maturin):
    with pytest.raises(ValueError, match="Python|maturin"):
        builder.validate_build_tools(checkout, python_version, maturin)


def test_reject_mixed_provider_pins(checkout):
    p = checkout / "rust/crates/innovate-core/Cargo.toml"
    p.write_text(
        p.read_text()
        + 'kairo-ecs-rng = { git = "https://github.com/edithatogo/kairos.git", rev = "'
        + "b" * 40
        + '" }\n'
    )
    with pytest.raises(ValueError, match="one immutable Kairos"):
        builder.kairos_revision(checkout)


def test_missing_provider_pin_is_not_an_attested_revision(checkout):
    (checkout / "rust/crates/innovate-core/Cargo.toml").write_text("[dependencies]\n")
    with pytest.raises(ValueError, match="one immutable Kairos"):
        builder.kairos_revision(checkout)


@pytest.mark.parametrize("revision", ["main", "ABCDEF", "a" * 39])
def test_reject_nonimmutable_provider_revision(checkout, revision):
    p = checkout / "rust/crates/innovate-core/Cargo.toml"
    p.write_text(p.read_text().replace("a1f6283575ad26918230adc122a99b5c08fc4da3", revision))
    with pytest.raises(ValueError, match="one immutable Kairos"):
        builder.kairos_revision(checkout)


def test_reject_new_untracked_source_during_build(checkout, tmp_path, monkeypatch):
    def drift(command, repository, log, environment):
        code = successful_build(command, repository, log, environment)
        (repository / "rust/new.rs").write_text("untracked during build")
        return code

    monkeypatch.setattr(builder, "run_build", drift)
    out = tmp_path / "evidence"
    with pytest.raises(ValueError, match="uncommitted build inputs"):
        builder.build(checkout, out, 1)
    assert not (out / "build-provenance.json").exists()


def test_builder_preserves_real_subprocess_exit_and_diagnostics(tmp_path):
    log_path = tmp_path / "build.log"
    with log_path.open("w") as log:
        code = builder.run_build(
            [sys.executable, "-c", "import sys; print('actual diagnostic'); sys.exit(19)"],
            tmp_path,
            log,
            os.environ.copy(),
        )
    assert code == 19
    assert log_path.read_text().strip() == "actual diagnostic"


def test_command_failure_returns_nonzero_without_success_json(checkout, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--repository", str(checkout), "--output", str(tmp_path / "out"), "--jobs", "0"]
    )
    assert builder.main() == 1
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "positive integer" in stderr


def test_command_success_reports_observed_artifact(checkout, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(builder, "run_build", successful_build)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--repository", str(checkout), "--output", str(tmp_path / "out")])
    assert builder.main() == 0
    stdout, stderr = capsys.readouterr()
    assert json.loads(stdout)["status"] == "built"
    assert stderr == ""
