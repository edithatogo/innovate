"""Offline checks for the six exact PBS objects and their receipt boundaries."""

import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "validation/data/pbs-sources.json"
SPEC = importlib.util.spec_from_file_location("pbs_source_manifest", ROOT / "scripts/validate_pbs_source_manifest.py")
assert SPEC is not None and SPEC.loader is not None
pbs_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pbs_manifest)
EXPECTED_OBJECTS = {
    "raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2023-24.XLSX": (
        "32ea64558d43531fed4bb395787bbc9279fdf45a2ddc6481ce8234aac676306b",
        372149,
    ),
    "raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2024-25.XLSX": (
        "147f4c5392e256d85beb3e47dfd8445085e5a8f0ecbbd330b535774d81aadf99",
        468764,
    ),
    "raw/pbs/utilisation/dos/dos-jul-2024-to-jun-2025-phrmcy-type.csv": (
        "4a961d1d128e3aa9449a1d3a38c27f1b742d1c13dc07d12553134e04e701a6a4",
        35384567,
    ),
    "raw/pbs/utilisation/dos/dos-jul-2025-to-jun-2026-phrmcy-type.csv": (
        "f502a61a472ef6db2986c5ca36a6f73b264cac1156830c38a5cf02f17bf59be8",
        36381946,
    ),
    "raw/pbs/utilisation/reference/pbs-item-drug-map.csv": (
        "51adf3c80df2c2d526bfe88d2d96199e545579ced1b32e7ba3550dfc15f358a8",
        1063694,
    ),
    "raw/pbs/utilisation/summary/dos-jul-2021-to-jun-2026.xlsx": (
        "48678536ddb71f9e7e4e6d654a0a90e6b242c22105e6864af1c6ef47318980ca",
        117376499,
    ),
}


@pytest.fixture
def manifest():
    return json.loads(MANIFEST.read_text())


def test_exact_frozen_manifest_and_rights(manifest):
    pbs_manifest.validate_manifest(manifest)
    assert manifest["source"] == {
        "dataset": "edithatogo/australian-pbs-utilisation-archive",
        "revision": "9f1d53cd50eea251b326da42a96f3ac72ebce97a",
    }
    objects = manifest["objects"]
    assert len(objects) == len(EXPECTED_OBJECTS) == len({obj["path"] for obj in objects})
    assert {obj["path"]: (obj["sha256"], obj["byte_count"]) for obj in objects} == EXPECTED_OBJECTS
    assert manifest["upstream"]["historical_rights"]["rights_state"] == "unknown"
    rights = manifest["upstream"]["rights_attestation"]
    assert rights["state"] == "maintainer_attested_exact_public_release_permission_for_pbs_utilisation_raw_object"
    assert rights["permission_instrument_independently_reviewed"] is False
    assert manifest["qualification"]["semantic_status"] == "not_qualified"
    assert manifest["qualification"]["raw_v4_admission"] == "six_raw_objects_admitted"
    assert manifest["qualification"]["raw_reader_canary"] == "six_passed"
    assert manifest["qualification"]["modelling_consumer_canary"] == "not_evidenced"


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("wrong_revision", "revision"),
        ("wrong_hash", "identity"),
        ("wrong_size", "identity"),
        ("duplicate_object", "identity"),
        ("missing_object", "identity"),
        ("historical_overwrite", "rights"),
        ("unsupported_clearance", "sensitivity"),
        ("false_semantic_qualification", "qualification"),
        ("false_canary", "qualification"),
        ("false_raw_canary", "qualification"),
        ("unknown_readiness_claim", "qualification"),
        ("bool_is_not_size", "identity"),
    ],
)
def test_manifest_rejects_changed_identity_or_unproved_claim(manifest, case, error):
    if case == "wrong_revision":
        manifest["source"]["revision"] = "0" * 40
    elif case == "wrong_hash":
        manifest["objects"][0]["sha256"] = "0" * 64
    elif case == "wrong_size":
        manifest["objects"][0]["byte_count"] += 1
    elif case == "duplicate_object":
        manifest["objects"][1] = copy.deepcopy(manifest["objects"][0])
    elif case == "missing_object":
        manifest["objects"].pop()
    elif case == "historical_overwrite":
        manifest["upstream"]["rights_attestation"]["state"] = "unknown"
    elif case == "unsupported_clearance":
        manifest["upstream"]["role_readiness"]["personal_data_absence_proven"] = True
    elif case == "false_semantic_qualification":
        manifest["qualification"]["semantic_status"] = "qualified"
    elif case == "false_canary":
        manifest["qualification"]["modelling_consumer_canary"] = "passed"
    elif case == "false_raw_canary":
        manifest["qualification"]["raw_reader_canary"] = "not_performed"
    elif case == "unknown_readiness_claim":
        manifest["qualification"]["model_ready"] = True
    elif case == "bool_is_not_size":
        manifest["objects"][0]["byte_count"] = True
    with pytest.raises(ValueError, match=error):
        pbs_manifest.validate_manifest(manifest)


@pytest.fixture
def replay_fixture(manifest):
    obj = copy.deepcopy(manifest["objects"][0])
    # Rebuild the deterministic replay identity; the source payload is not bundled.
    raw = (ROOT / "tests/validation/fixtures/pbs-replay-40677682866.json").read_bytes()
    receipt = json.loads(raw)
    registration = receipt["replay"]["registration"]
    assert registration["idempotency_key"] is None
    identity = hashlib.sha256(
        json.dumps([registration["partition_id"], registration["snapshot_id"]], separators=(",", ":")).encode()
    ).hexdigest()
    placeholder = b'"idempotency_key": null'
    assert raw.count(placeholder) == 1
    raw = raw.replace(placeholder, f'"idempotency_key": "{identity}"'.encode())
    assert hashlib.sha256(raw).hexdigest() == obj["receipt"]["sha256"]
    receipt = json.loads(raw)
    return obj, receipt, raw


def test_matching_local_receipt_fixture_is_accepted(manifest, replay_fixture):
    obj, _, raw = replay_fixture
    pbs_manifest.validate_replay_receipt(raw, manifest, obj)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("revision", "0" * 40, "revision"),
        ("sha256", "0" * 64, "sha256"),
        ("byte_count", 1, "byte_count"),
        ("path", "raw/other", "path"),
        ("dataset", "other/dataset", "dataset"),
    ],
)
def test_receipt_scope_mismatch_is_rejected(manifest, replay_fixture, field, value, error):
    obj, receipt, _ = replay_fixture
    receipt["scope"][field] = value
    raw = json.dumps(receipt, sort_keys=True).encode()
    with pytest.raises(ValueError, match=error):
        pbs_manifest.validate_replay_receipt(raw, manifest, obj)


def test_changed_receipt_bytes_are_rejected_even_with_same_json(manifest, replay_fixture):
    obj, _, raw = replay_fixture
    with pytest.raises(ValueError, match="receipt.*sha256"):
        pbs_manifest.validate_replay_receipt(raw + b" ", manifest, obj)


def test_duplicate_receipt_keys_are_rejected(manifest, replay_fixture):
    obj, _, raw = replay_fixture
    duplicate = raw.replace(b"{", b'{"status":"passed",', 1)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        pbs_manifest.validate_replay_receipt(duplicate, manifest, obj)


def test_changed_upstream_evidence_file_is_rejected(manifest, tmp_path):
    item = manifest["upstream"]["historical_rights"]
    file = tmp_path / item["path"]
    file.parent.mkdir(parents=True)
    file.write_bytes(b"altered metadata")
    with pytest.raises(ValueError, match="historical_rights sha256"):
        pbs_manifest.validate_upstream_files(manifest, tmp_path)


def test_receipt_cannot_supply_its_own_replacement_digest(manifest, replay_fixture):
    obj, _, raw = replay_fixture
    changed = raw + b" "
    obj["receipt"]["sha256"] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="object identity"):
        pbs_manifest.validate_replay_receipt(changed, manifest, obj)


def test_registration_must_match_payload_and_later_rights(manifest, replay_fixture):
    obj, receipt, _ = replay_fixture
    receipt["replay"]["registration"]["rights_state"] = "unknown"
    with pytest.raises(ValueError, match="registration rights"):
        pbs_manifest.validate_replay_receipt(json.dumps(receipt).encode(), manifest, obj)

    receipt["replay"]["registration"]["rights_state"] = (
        "maintainer_attested_exact_public_release_permission_for_pbs_utilisation_raw_object"
    )
    receipt["replay"]["registration"]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registration payload sha256"):
        pbs_manifest.validate_replay_receipt(json.dumps(receipt).encode(), manifest, obj)


def test_byte_verification_rejects_corruption_and_size_changes():
    raw = b"synthetic source bytes only"
    digest = hashlib.sha256(raw).hexdigest()
    pbs_manifest.validate_payload(io.BytesIO(raw), digest, len(raw))
    with pytest.raises(ValueError, match="sha256"):
        pbs_manifest.validate_payload(io.BytesIO(raw[:-1] + b"!"), digest, len(raw))
    with pytest.raises(ValueError, match="size"):
        pbs_manifest.validate_payload(io.BytesIO(raw + b"!"), digest, len(raw))


def test_source_payload_requires_the_frozen_revision_and_path(manifest):
    raw = io.BytesIO(b"not a real PBS object")
    with pytest.raises(ValueError, match="revision"):
        pbs_manifest.validate_source_payload(raw, manifest, manifest["objects"][0]["path"], "0" * 40)
    with pytest.raises(ValueError, match="identity"):
        pbs_manifest.validate_source_payload(raw, manifest, "raw/other", manifest["source"]["revision"])
    assert raw.tell() == 0


def test_source_payload_rejects_changed_bytes_for_a_known_object(manifest):
    with pytest.raises(ValueError, match="size"):
        pbs_manifest.validate_source_payload(
            io.BytesIO(b"not a real PBS object"),
            manifest,
            manifest["objects"][0]["path"],
            manifest["source"]["revision"],
        )


@pytest.mark.parametrize("mutation", ["root", "source", "upstream", "object"])
def test_manifest_rejects_missing_or_unknown_fields(manifest, mutation):
    if mutation == "root":
        manifest["unexpected"] = True
    elif mutation == "source":
        del manifest["source"]["dataset"]
    elif mutation == "upstream":
        del manifest["upstream"]["rights_attestation"]
    else:
        manifest["objects"][0]["unexpected"] = True
    with pytest.raises(ValueError):
        pbs_manifest.validate_manifest(manifest)


@pytest.mark.parametrize(
    ("raw", "message"), [(b"[", "invalid JSON"), (b"NaN", "invalid JSON constant"), (b"[]", "not an object")]
)
def test_replay_receipt_rejects_malformed_json(manifest, replay_fixture, raw, message):
    obj, _, _ = replay_fixture
    with pytest.raises(ValueError, match=message):
        pbs_manifest.validate_replay_receipt(raw, manifest, obj)


@pytest.mark.parametrize(
    ("field", "message"),
    [("scope", "scope is invalid"), ("hosted_workflow", "workflow is invalid"), ("replay", "registration is invalid")],
)
def test_replay_receipt_rejects_missing_nested_objects(manifest, replay_fixture, field, message):
    obj, receipt, _ = replay_fixture
    receipt[field] = None
    with pytest.raises(ValueError, match=message):
        pbs_manifest.validate_replay_receipt(json.dumps(receipt).encode(), manifest, obj)


@pytest.mark.parametrize(("digest", "size"), [("bad", 1), ("0" * 64, -1), ("0" * 64, True)])
def test_payload_rejects_invalid_expectations(digest, size):
    with pytest.raises(ValueError):
        pbs_manifest.validate_payload(io.BytesIO(b"x"), digest, size)


def test_upstream_evidence_path_cannot_leave_supplied_root(tmp_path):
    with pytest.raises(ValueError, match="remain under"):
        pbs_manifest._read_under(tmp_path, "../outside")


def test_cli_validates_frozen_manifest_without_claiming_model_readiness(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pbs-manifest", "--manifest", str(MANIFEST)])
    pbs_manifest.main()
    output = capsys.readouterr().out
    assert "metadata only" in output
    assert "semantic and modelling qualification remain open" in output


def test_cli_rejects_invalid_manifest(tmp_path, monkeypatch, capsys):
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text("not json")
    monkeypatch.setattr(sys, "argv", ["pbs-manifest", "--manifest", str(bad_manifest)])
    with pytest.raises(SystemExit) as exc:
        pbs_manifest.main()
    assert exc.value.code == 1
    assert "PBS manifest validation failed" in capsys.readouterr().err
