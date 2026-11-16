"""The manuscript preflight detects drift without inventing submission readiness."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRACK = "conductor/tracks/research_software_registry_readiness_20260721"
SNAPSHOT = f"{TRACK}/joss/source-snapshot.json"
PAPER = f"{TRACK}/joss/paper.md"
CLAIMS = f"{TRACK}/joss/claim-evidence.md"
SCRIPT = ROOT / "scripts/check_joss_readiness.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("joss_readiness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packet(tmp_path):
    """Copy the independently reviewed edition, not a synthetic ready paper."""
    data = json.loads((ROOT / SNAPSHOT).read_text(encoding="utf-8"))
    for name in [*data["source_sha256"], *data["edition_sha256"]]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Git may materialize CRLF on Windows; synthetic mutations start from
        # the reviewed LF bytes on every host. The real-record test stays raw.
        target.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return data


def replace_input(root, packet, name, before, after, group="edition_sha256"):
    path = root / name
    content = path.read_text(encoding="utf-8")
    assert before in content
    path.write_text(content.replace(before, after), encoding="utf-8", newline="\n")
    packet[group][name] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_draft_is_valid_but_not_submission_ready(checker):
    packet = checker.read_snapshot(ROOT / SNAPSHOT)
    report = checker.validate_packet(ROOT, packet)
    assert report["snapshot_valid"] is True
    assert report["submission_ready"] is False
    assert report["source_files_checked"] == 14
    assert report["edition_files_checked"] == 3
    assert report["required_sections_checked"] == 6
    assert 750 <= report["prose_word_count"] <= 1750
    assert report["git_revision_verified"] is False
    assert report["scientific_qualification_performed"] is False
    assert report["online_verification_performed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("state", "approved", "unapproved"),
        ("submission_ready", True, "submission_ready"),
        ("submission_ready", "false", "submission_ready"),
        ("software_revision", "main", "revision"),
        ("software_revision", "0" * 40, "revision"),
        ("software_version", "0.5.0", "version"),
        ("source_sha256", [], "source_sha256"),
        ("edition_sha256", None, "edition_sha256"),
    ],
)
def test_invalid_identity_and_readiness_are_rejected(checker, packet, tmp_path, field, value, message):
    packet[field] = value
    with pytest.raises(ValueError, match=message):
        checker.validate_packet(tmp_path, packet)


@pytest.mark.parametrize("group", ["source_sha256", "edition_sha256"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "empty", "bad_digest", "missing_file"])
def test_incomplete_or_stale_fingerprints_fail(checker, packet, tmp_path, group, mutation):
    name = next(iter(packet[group]))
    if mutation == "missing":
        del packet[group][name]
    elif mutation == "extra":
        packet[group]["../private.txt"] = "a" * 64
    elif mutation == "changed":
        (tmp_path / name).write_bytes(b"Changed after the edition was reviewed.\n")
    elif mutation == "empty":
        (tmp_path / name).write_bytes(b"\n")
    elif mutation == "bad_digest":
        packet[group][name] = "not-a-digest"
    else:
        (tmp_path / name).unlink()
    with pytest.raises(ValueError):
        checker.validate_packet(tmp_path, packet)


def test_duplicate_json_fields_are_not_silently_overwritten(checker, tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"submission_ready":true,"submission_ready":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        checker.read_snapshot(path)


def test_nonobject_snapshot_is_rejected(checker, tmp_path):
    with pytest.raises(ValueError, match="object"):
        checker.validate_packet(tmp_path, [])


def test_symlink_cannot_satisfy_source_evidence_outside_root(checker, packet, tmp_path):
    path = tmp_path / "README.md"
    outside = tmp_path.parent / "unrelated-joss-source.md"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Platform cannot create a test symlink: {error}")
    with pytest.raises(ValueError, match="within repository"):
        checker.validate_packet(tmp_path, packet)


def test_windows_checkout_line_endings_are_explicitly_reported(checker, packet, tmp_path):
    path = tmp_path / PAPER
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    report = checker.validate_packet(tmp_path, packet)
    assert report["normalized_text_paths"] == [PAPER]
    assert report["submission_ready"] is False


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ("# Research impact statement", "# Intended future use", "section"),
        ("# Summary", "# Summary\n\n# Summary", "section"),
        ("# Summary", "> # Summary", "section"),
        ("# Summary", "```markdown\n# Summary\n```", "section"),
        ("# Summary", "<!--\n# Summary\n-->", "section"),
        ("# Summary", "# Software design", "section"),
        ("# References", "# Sources", "References"),
        ("not author-approved or submission-ready", "approved for submission", "draft notice"),
    ],
)
def test_rehashed_structural_regressions_still_fail(checker, packet, tmp_path, before, after, message):
    replace_input(tmp_path, packet, PAPER, before, after)
    with pytest.raises(ValueError, match=message):
        checker.validate_packet(tmp_path, packet)


@pytest.mark.parametrize("name", [PAPER, CLAIMS])
def test_paper_and_claim_register_must_identify_the_recorded_revision(checker, packet, tmp_path, name):
    replace_input(tmp_path, packet, name, packet["software_revision"], "a" * 40)
    with pytest.raises(ValueError, match="revision"):
        checker.validate_packet(tmp_path, packet)


def test_rehashed_project_version_change_requires_a_new_edition(checker, packet, tmp_path):
    replace_input(tmp_path, packet, "pyproject.toml", 'version = "0.7.0a1"', 'version = "0.8.0a1"', "source_sha256")
    with pytest.raises(ValueError, match="version"):
        checker.validate_packet(tmp_path, packet)


@pytest.mark.parametrize("version", ["", "0.7"])
def test_empty_or_prefix_version_cannot_match_a_different_edition(checker, packet, tmp_path, version):
    replace_input(tmp_path, packet, "pyproject.toml", 'version = "0.7.0a1"', f'version = "{version}"', "source_sha256")
    packet["software_version"] = version
    with pytest.raises(ValueError, match="version"):
        checker.validate_packet(tmp_path, packet)


def test_rehashed_empty_required_section_fails(checker, packet, tmp_path):
    paper = (tmp_path / PAPER).read_text(encoding="utf-8")
    start = paper.index("# Statement of need")
    end = paper.index("# State of the field")
    replace_input(tmp_path, packet, PAPER, paper[start:end], "# Statement of need\n\n")
    with pytest.raises(ValueError, match="empty"):
        checker.validate_packet(tmp_path, packet)


def test_unterminated_comment_cannot_hide_remaining_manuscript(checker, packet, tmp_path):
    replace_input(tmp_path, packet, PAPER, "# Statement of need", "<!--\n# Statement of need")
    with pytest.raises(ValueError, match="comment"):
        checker.validate_packet(tmp_path, packet)


def test_hidden_draft_notice_is_rejected(checker, packet, tmp_path):
    notice = "> Working draft: not author-approved or submission-ready."
    replace_input(tmp_path, packet, PAPER, notice, f"<!-- {notice} -->")
    with pytest.raises(ValueError, match="draft notice"):
        checker.validate_packet(tmp_path, packet)


def test_comment_syntax_inside_code_is_not_a_prose_comment(checker, packet, tmp_path):
    original = checker.validate_packet(tmp_path, packet)["prose_word_count"]
    replace_input(tmp_path, packet, PAPER, "# Summary", "```html\n<!--\n```\n\n# Summary")
    assert checker.validate_packet(tmp_path, packet)["prose_word_count"] == original


@pytest.mark.parametrize("project", ['"not a table"', "[]"])
def test_valid_toml_with_invalid_project_shape_returns_json_error(checker, packet, tmp_path, capsys, project):
    path = tmp_path / "pyproject.toml"
    path.write_text(f"project = {project}\n", encoding="utf-8")
    packet["source_sha256"]["pyproject.toml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / SNAPSHOT).write_text(json.dumps(packet), encoding="utf-8")
    assert checker.main(["--root", str(tmp_path), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["snapshot_valid"] is False
    assert "project" in report["error"]


def test_word_budget_excludes_frontmatter_comments_and_code(checker, packet, tmp_path):
    original = checker.validate_packet(tmp_path, packet)["prose_word_count"]
    excluded = "ignored " * 1800
    replace_input(
        tmp_path,
        packet,
        PAPER,
        "# Summary\n",
        f"# Summary\n\n<!-- {excluded} -->\n```text\n{excluded}\n```\n",
    )
    assert checker.validate_packet(tmp_path, packet)["prose_word_count"] == original


def test_rehashed_overlength_manuscript_fails(checker, packet, tmp_path):
    replace_input(tmp_path, packet, PAPER, "# References", ("additional " * 1800) + "\n\n# References")
    with pytest.raises(ValueError, match="word"):
        checker.validate_packet(tmp_path, packet)


def test_cli_reports_valid_pending_and_rejects_submission_requirement(tmp_path):
    for extra, expected in (([], 0), (["--require-submission-ready"], 1)):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(ROOT), "--json", *extra],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected, result.stderr
        report = json.loads(result.stdout)
        assert report["snapshot_valid"] is True
        assert report["submission_ready"] is False


def test_cli_missing_snapshot_returns_actionable_json(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["snapshot_valid"] is False
    assert "error" in report
