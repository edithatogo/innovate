"""Incomplete native support cannot silently retire an existing feature."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_feature",
        "drop_model",
        "make_optional",
        "duplicate_feature",
        "missing_evidence",
        "missing_evidence_file",
        "outside_evidence_root",
        "missing_binding",
        "unsupported_binding",
        "empty_evidence",
        "changed_qualified_evidence",
        "invalid_status",
        "qualified_with_remaining_work",
        "nonboolean_release",
        "premature_release",
    ],
)
def test_migration_policy_rejects_feature_loss(tmp_path, monkeypatch, mutation):
    spec = importlib.util.spec_from_file_location("migration_policy", ROOT / "scripts/check_migration.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ledger = json.loads((ROOT / "migration-ledger.json").read_text())
    for entry in ("python/innovate/__init__.py", "julia/Project.toml", "r/DESCRIPTION"):
        path = tmp_path / "bindings" / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if mutation in {"nonboolean_release", "premature_release"}:
        ledger["release_ready"] = "false" if mutation == "nonboolean_release" else True
        for group in ("models", "feature_areas"):
            for item in ledger[group]:
                if evidence := item.get("evidence"):
                    target = tmp_path / evidence
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((ROOT / evidence).read_bytes())
    elif mutation == "drop_feature":
        ledger["feature_areas"].pop()
    elif mutation == "drop_model":
        ledger["models"].pop()
    elif mutation == "make_optional":
        ledger["feature_areas"][0]["required"] = False
    elif mutation == "missing_binding":
        (tmp_path / "bindings/r/DESCRIPTION").unlink()
    elif mutation == "unsupported_binding":
        path = tmp_path / "bindings/java"
        path.mkdir()
        (path / "pom.xml").touch()
    elif mutation == "missing_evidence":
        del ledger["models"][0]["evidence"]
    elif mutation == "missing_evidence_file":
        ledger["models"][0]["evidence"] = "missing.md"
    elif mutation == "outside_evidence_root":
        outside = tmp_path.parent / "outside.md"
        outside.write_text("external evidence")
        ledger["models"][0]["evidence"] = "../outside.md"
    elif mutation == "empty_evidence":
        ledger["models"][0].update(status="qualified", evidence="empty.md")
        (tmp_path / "empty.md").write_text(" \n")
    elif mutation == "changed_qualified_evidence":
        ledger["models"][0].update(status="qualified", evidence="receipt.md", evidence_sha256="0" * 64)
        (tmp_path / "receipt.md").write_text("changed receipt")
    elif mutation == "qualified_with_remaining_work":
        ledger["models"][0].update(status="qualified", remaining="Scientific parity open")
    elif mutation == "invalid_status":
        ledger["models"][0]["status"] = "complete"
    else:
        ledger["feature_areas"].append(ledger["feature_areas"][0])
    (tmp_path / "migration-ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["check_migration.py"])
    expected = (
        "(lost|required|Missing migration evidence|Invalid migration evidence|Empty migration evidence|"
        "Invalid migration status|release_ready|Missing required binding|Unsupported language binding)"
    )
    if mutation == "qualified_with_remaining_work":
        expected = "Qualified migration obligation still has remaining work"
    if mutation == "changed_qualified_evidence":
        expected = "Qualified migration evidence digest is missing or mismatched"
    with pytest.raises(ValueError, match=expected):
        module.main()


def test_release_source_digest_tracks_source_and_excludes_receipts(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("migration_policy", ROOT / "scripts/check_migration.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def git(*args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init")
    source = tmp_path / "Cargo.toml"
    source.write_text("first")
    git("add", "Cargo.toml")
    git("commit", "-m", "First source")
    first = module.release_source_digest()
    (tmp_path / "migration-ledger.json").write_text("{}")
    assert module.release_source_digest() == first

    source.write_text("second")
    with pytest.raises(ValueError, match="clean source inputs"):
        module.release_source_digest()
    git("add", "Cargo.toml")
    git("commit", "-m", "Second source")
    assert module.release_source_digest() != first

    for entry in ("python/innovate/__init__.py", "julia/Project.toml", "r/DESCRIPTION"):
        path = tmp_path / "bindings" / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    git("add", "bindings", "pyproject.toml")
    git("commit", "-m", "Add binding inputs")

    receipt = b"synthetic qualification evidence"
    (tmp_path / "receipt.md").write_bytes(receipt)
    ledger = json.loads((ROOT / "migration-ledger.json").read_text())
    for group in ("models", "feature_areas"):
        for item in ledger[group]:
            item.pop("remaining", None)
            item.update(status="qualified", evidence="receipt.md", evidence_sha256=hashlib.sha256(receipt).hexdigest())
    ledger.update(release_ready=True, qualified_source_sha256=module.release_source_digest())
    (tmp_path / "migration-ledger.json").write_text(json.dumps(ledger))
    monkeypatch.setattr("sys.argv", ["check_migration.py", "--release"])
    assert module.main() == 0

    source.write_text("third")
    git("add", "Cargo.toml")
    git("commit", "-m", "Change release source")
    with pytest.raises(ValueError, match="does not match the release source digest"):
        module.main()
