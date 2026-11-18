"""Renderer guard tests use synthetic processes, never publication evidence."""

import importlib.util
import json
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "conductor/tracks/research_software_registry_readiness_20260721/joss/render.py"
NOTICE = "Working draft: not author-approved or submission-ready."
DRAFT_TEX = "\\usepackage{draftwatermark}\n\\usepackage{lineno}\n" + NOTICE


@pytest.fixture
def render_module():
    spec = importlib.util.spec_from_file_location("joss_render", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def renderer(render_module, tmp_path, monkeypatch):
    module = render_module
    # Subprocesses below model a POSIX Make environment while fixture files use
    # the host filesystem. Test the real POSIX path guard independently rather
    # than making Windows CI's temporary paths fail every orchestration case.
    monkeypatch.setattr(module, "safe_make_path", lambda path: path.resolve())
    packet = tmp_path / "repository/conductor/tracks/test/joss"
    packet.mkdir(parents=True)
    (packet / "paper.md").write_text("SYNTHETIC TEST INPUT\n" + NOTICE, encoding="utf-8")
    (packet / "paper.bib").write_text("@article{fixture, title={Synthetic}}", encoding="utf-8")
    (packet / "source-snapshot.json").write_text(json.dumps({"software_revision": "b" * 40}))
    monkeypatch.setattr(module, "PACKET", packet)
    state = SimpleNamespace(
        module=module,
        packet=packet,
        inara=tmp_path / "inara",
        output=tmp_path / "output",
        pandoc=tmp_path / "bin/pandoc",
        revision=module.INARA_REVISION,
        pandoc_version="pandoc 3.6.4",
        checkout_status="",
        integrity_exit=0,
        pdf_exit=2,
        tex=DRAFT_TEX,
        mutation=None,
        calls=[],
    )

    def read_command(command, cwd=None):
        if command == ["git", "rev-parse", "HEAD"]:
            return state.revision if cwd == state.inara else "b" * 40
        if command == ["git", "status", "--porcelain"]:
            return state.checkout_status
        if command == [str(state.pandoc), "--version"]:
            return state.pandoc_version
        raise AssertionError(command)

    def which(command, **kwargs):
        return str(state.pandoc) if command == "pandoc" else "/usr/bin/" + command

    def run(command, **kwargs):
        state.calls.append(command)
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "SYNTHETIC tool version", "")
        if command[1].endswith("check_joss_readiness.py"):
            kwargs["stdout"].write("SYNTHETIC integrity-check process\n")
            return subprocess.CompletedProcess(command, state.integrity_exit)
        if "--from=bibtex" in command:
            destination = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
            destination.write_text(json.dumps([{"id": "fixture"}]))
            return subprocess.CompletedProcess(command, 0)
        assert command[0] == "make", command
        destination = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("TARGET_FOLDER=")))
        if command[1] == "tex":
            (destination / "paper.tex").write_text(state.tex, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)
        assert command[1] == "pdf", command
        if state.mutation == "change":
            (packet / "paper.md").write_text("CHANGED SYNTHETIC INPUT", encoding="utf-8")
        elif state.mutation == "delete":
            (packet / "paper.md").unlink()
        if state.pdf_exit:
            kwargs["stderr"].write("! SYNTHETIC PDF FAILURE: missing fixture dependency\n")
        else:
            (destination / "paper.pdf").write_bytes(b"SYNTHETIC OUTPUT; NOT A PDF OR ACCEPTANCE EVIDENCE")
        return subprocess.CompletedProcess(command, state.pdf_exit)

    monkeypatch.setattr(module, "read_command", read_command)
    monkeypatch.setattr(module.shutil, "which", which)
    monkeypatch.setattr(module.subprocess, "run", run)
    state.argv = ["--inara", str(state.inara), "--output-dir", str(state.output), "--source-date-epoch", "1794787200"]
    return state


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "a" * 40),
        ("pandoc_version", "pandoc 3.1.11.1"),
        ("checkout_status", " M data/templates/default.latex"),
    ],
)
def test_version_and_checkout_guards_reject_before_output(renderer, field, value):
    setattr(renderer, field, value)
    with pytest.raises(SystemExit) as failure:
        renderer.module.main(renderer.argv)
    assert failure.value.code == 2
    assert not renderer.output.exists()
    assert renderer.calls == []


def test_posix_make_path_text_is_accepted_independent_of_host(render_module):
    path = PurePosixPath("/tmp/joss-draft_20261116/paper.md")
    assert render_module.safe_make_path(SimpleNamespace(resolve=lambda: path)) == path


@pytest.mark.parametrize(
    "path",
    [
        PurePosixPath("/tmp/unsafe path"),
        PurePosixPath("/tmp/unsafe;path"),
        PurePosixPath("/tmp/unsafe$path"),
        PureWindowsPath(r"C:\inara\paper.md"),
        PureWindowsPath(r"\\server\share\paper.md"),
    ],
)
def test_make_path_text_rejects_metacharacters_and_windows_paths(render_module, path):
    with pytest.raises(ValueError, match="spaces/metacharacters"):
        render_module.safe_make_path(SimpleNamespace(resolve=lambda: path))


def test_existing_receipt_is_never_overwritten(renderer):
    renderer.output.mkdir()
    receipt = renderer.output / "receipt.json"
    receipt.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        renderer.module.main(renderer.argv)
    assert failure.value.code == 2
    assert receipt.read_text(encoding="utf-8") == "prior evidence"


def test_pdf_failure_retains_stages_and_forces_draft_without_visual_approval(renderer):
    assert renderer.module.main(renderer.argv) == 1
    report = json.loads((renderer.output / "receipt.json").read_text())
    assert report["status"] == "blocked"
    assert report["checks"]["source_integrity"]["status"] == "passed"
    assert report["checks"]["bibtex_parse"]["entry_ids"] == ["fixture"]
    assert report["checks"]["official_tex"]["status"] == "passed"
    assert report["checks"]["draft_tex_markers"]["status"] == "passed"
    assert report["checks"]["official_pdf"] == {"status": "failed", "exit_code": 2}
    assert report["visual_review"] == "not_performed"
    assert report["submission_ready"] is False
    assert report["scientific_or_release_qualification"] is False
    assert report["diagnostic_lines"] == ["! SYNTHETIC PDF FAILURE: missing fixture dependency"]
    assert not (renderer.output / "paper.pdf").exists()
    for command in (call for call in renderer.calls if call[0] == "make"):
        options = next(arg for arg in command if arg.startswith("PANDOC="))
        assert "--metadata=draft:true" in options
        assert "--variable=draft:true" in options
        assert "--fail-if-warnings" in options


def test_failed_integrity_prevents_parsing_and_rendering(renderer):
    renderer.integrity_exit = 1
    assert renderer.module.main(renderer.argv) == 1
    report = json.loads((renderer.output / "receipt.json").read_text())
    assert report["status"] == "blocked"
    assert list(report["checks"]) == ["source_integrity"]
    assert not (renderer.output / "paper.tex").exists()


@pytest.mark.parametrize("missing", ["\\usepackage{draftwatermark}", "\\usepackage{lineno}", NOTICE])
def test_missing_draft_markers_prevent_pdf_attempt(renderer, missing):
    renderer.tex = DRAFT_TEX.replace(missing, "")
    assert renderer.module.main(renderer.argv) == 1
    report = json.loads((renderer.output / "receipt.json").read_text())
    assert report["status"] == "blocked"
    assert "draft" in report["error"]
    assert "official_pdf" not in report["checks"]


@pytest.mark.parametrize("mutation", ["change", "delete"])
def test_changed_or_missing_input_invalidates_even_synthetic_success(renderer, mutation):
    renderer.pdf_exit = 0
    renderer.mutation = mutation
    assert renderer.module.main(renderer.argv) == 1
    report = json.loads((renderer.output / "receipt.json").read_text())
    assert report["status"] == "blocked"
    assert "inputs changed" in report["error"]
    assert report["visual_review"] == "not_performed"
    if mutation == "delete":
        assert "input_recheck_error" in report


def test_synthetic_process_success_still_requires_visual_and_submission_review(renderer):
    renderer.pdf_exit = 0
    assert renderer.module.main(renderer.argv) == 0
    report = json.loads((renderer.output / "receipt.json").read_text())
    assert report["status"] == "rendered_draft_visual_review_pending"
    assert report["visual_review"] == "not_performed"
    assert report["submission_ready"] is False
    assert report["scientific_or_release_qualification"] is False
