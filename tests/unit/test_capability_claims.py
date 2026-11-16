"""Fail-closed tests for capability evidence and source identity."""

import copy
import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("capability_claims", ROOT / "scripts/validate_capability_claims.py")
assert SPEC is not None and SPEC.loader is not None
claims = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(claims)


@pytest.fixture
def invented_claim():
    return {
        "schema_version": "1.0",
        "revision": "1" * 40,
        "capabilities": [
            {
                "model": "invented",
                "operation": "fit",
                "payload": "x",
                "backend": "rust_native",
                "evidence": {
                    "native_receipt": "receipt.json",
                    "scientific_receipt": None,
                    "empirical_receipt": None,
                    "release_receipt": None,
                },
            }
        ],
    }


def test_checked_in_native_claims_are_not_qualified_by_source_paths():
    data = json.loads((ROOT / "conductor/capabilities.json").read_text())
    errors = claims.validate_capabilities(data, expected_revision=data["revision"], root=ROOT)
    assert any("native_receipt" in error and "source-bound" in error for error in errors)
    assert any("scientific_receipt" in error and "source-bound" in error for error in errors)


def test_receipt_with_self_declared_success_is_not_proof(invented_claim, tmp_path):
    (tmp_path / "receipt.json").write_text(json.dumps({"status": "passed", "revision": "1" * 40}))
    errors = claims.validate_capabilities(invented_claim, expected_revision="1" * 40, root=tmp_path)
    assert any("native_receipt" in error and "source-bound" in error for error in errors)


@pytest.mark.parametrize(("path", "message"), [(None, "missing"), ("missing.json", "does not exist")])
def test_missing_native_receipts_remain_rejected(invented_claim, tmp_path, path, message):
    invented_claim["capabilities"][0]["evidence"]["native_receipt"] = path
    errors = claims.validate_capabilities(invented_claim, expected_revision=invented_claim["revision"], root=tmp_path)
    assert any("native_receipt" in error and message in error for error in errors)


def test_absolute_or_traversing_receipt_cannot_escape_checkout(invented_claim, tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}-outside.json"
    external.write_text("{}")
    for path in (str(external), "../outside.json"):
        data = copy.deepcopy(invented_claim)
        data["capabilities"][0]["evidence"]["native_receipt"] = path
        errors = claims.validate_capabilities(data, expected_revision=data["revision"], root=tmp_path)
        assert any("native_receipt" in error and "outside" in error for error in errors)


def test_symlink_escape_is_rejected_when_supported(invented_claim, tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}-outside.json"
    external.write_text("{}")
    try:
        (tmp_path / "link").symlink_to(external)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    invented_claim["capabilities"][0]["evidence"]["native_receipt"] = "link"
    errors = claims.validate_capabilities(invented_claim, expected_revision=invented_claim["revision"], root=tmp_path)
    assert any("native_receipt" in error and "outside" in error for error in errors)


def test_invalid_receipt_path_is_rejected(invented_claim, tmp_path):
    invented_claim["capabilities"][0]["evidence"]["native_receipt"] = "invalid\x00path"
    errors = claims.validate_capabilities(invented_claim, expected_revision=invented_claim["revision"], root=tmp_path)
    assert any("native_receipt" in error and "invalid" in error for error in errors)


def test_arbitrary_backend_and_invalid_capability_are_not_silently_accepted(invented_claim, tmp_path):
    data = copy.deepcopy(invented_claim)
    data["capabilities"][0]["backend"] = "unsupported_engine"
    errors = claims.validate_capabilities(data, expected_revision=data["revision"], root=tmp_path)
    assert any("unsupported" in error for error in errors)
    data["capabilities"] = ["not an object"]
    assert claims.validate_capabilities(data, expected_revision=data["revision"], root=tmp_path)


def test_stale_or_unverified_checkout_revision_is_rejected(invented_claim, tmp_path):
    errors = claims.validate_capabilities(invented_claim, expected_revision="1" * 40, root=tmp_path)
    assert any("checkout revision" in error for error in errors)
    data = json.loads((ROOT / "conductor/capabilities.json").read_text())
    errors = claims.validate_capabilities(data, expected_revision="0" * 40, root=ROOT)
    assert any("revision" in error for error in errors)


def test_historical_source_commit_can_be_named_without_claiming_qualification(invented_claim, tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    author = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid")
    git(*author, "commit", "--allow-empty", "-qm", "source")
    historical_revision = git("rev-parse", "HEAD").strip()
    git(*author, "commit", "--allow-empty", "-qm", "later")
    (tmp_path / "receipt.json").write_text("{}")
    invented_claim["revision"] = historical_revision
    errors = claims.validate_capabilities(invented_claim, expected_revision=historical_revision, root=tmp_path)
    assert not any("revision" in error for error in errors)
    assert any("source-bound" in error for error in errors)


def test_nonexistent_source_commit_cannot_be_verified(invented_claim):
    errors = claims.validate_capabilities(invented_claim, expected_revision="1" * 40, root=ROOT)
    assert any("revision" in error for error in errors)
