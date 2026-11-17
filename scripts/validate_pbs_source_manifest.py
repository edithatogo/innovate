"""Validate a frozen PBS manifest and existing Atlas metadata, without network access.

This verifies exact metadata and optional bytes only. It does not qualify source
semantics, patient-level interpretation, modelling readiness or a licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import typing

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = "edithatogo/australian-pbs-utilisation-archive"
REVISION = "9f1d53cd50eea251b326da42a96f3ac72ebce97a"
UPSTREAM_COMMIT = "c6d289685ff5712c64e2ab62a6974e20ddaf4712"
RECEIPT_BASE = "quality/qualifications/australian-m112-pbs-v3-replays-20261028"

# Anchored in conductor/research/medicines-utilisation-20261107.md and the
# source-owner's immutable role-readiness and replay records, not in filenames.
OBJECTS = {
    "raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2023-24.XLSX": (
        "32ea64558d43531fed4bb395787bbc9279fdf45a2ddc6481ce8234aac676306b",
        372149,
        "aggregate_annual_expenditure_workbook",
        40677682866,
        "18f0101937e34aaa10f19387cd1d5b1fcc67933bd9bf4928ca9fab561d89754a",
    ),
    "raw/pbs/expenditure/Expenditure-prescriptions-report-tables-2024-25.XLSX": (
        "147f4c5392e256d85beb3e47dfd8445085e5a8f0ecbbd330b535774d81aadf99",
        468764,
        "aggregate_annual_expenditure_workbook",
        40678419501,
        "499ccc5f6a22f9d34da12504466b4246a14eb1228a31796bc8adf5586f40f95d",
    ),
    "raw/pbs/utilisation/dos/dos-jul-2024-to-jun-2025-phrmcy-type.csv": (
        "4a961d1d128e3aa9449a1d3a38c27f1b742d1c13dc07d12553134e04e701a6a4",
        35384567,
        "aggregate_date_of_supply_supplementary_csv",
        40678703243,
        "f810698bea1996661b412d76e09b9f908a1631f20d76b2a4d34f633535f53616",
    ),
    "raw/pbs/utilisation/dos/dos-jul-2025-to-jun-2026-phrmcy-type.csv": (
        "f502a61a472ef6db2986c5ca36a6f73b264cac1156830c38a5cf02f17bf59be8",
        36381946,
        "aggregate_date_of_supply_supplementary_csv",
        40679077698,
        "7ff834aa16bf056f7120b00d7e30ea5f978443619b0198ec8a07bbfe547403f6",
    ),
    "raw/pbs/utilisation/reference/pbs-item-drug-map.csv": (
        "51adf3c80df2c2d526bfe88d2d96199e545579ced1b32e7ba3550dfc15f358a8",
        1063694,
        "item_drug_reference_map",
        40679177671,
        "121efc1d1b1f02185af6bab0d9badd0be990c2389b835c7dc0b4d8be965170c4",
    ),
    "raw/pbs/utilisation/summary/dos-jul-2021-to-jun-2026.xlsx": (
        "48678536ddb71f9e7e4e6d654a0a90e6b242c22105e6864af1c6ef47318980ca",
        117376499,
        "aggregate_multiyear_date_of_supply_workbook",
        40679271451,
        "e035b6fea7dd5dd581abd612b8880e53fafd43841040349116948dec18554bef",
    ),
}
UPSTREAM = {
    "repository": "edithatogo/global-medicines-atlas",
    "commit": UPSTREAM_COMMIT,
    "historical_rights": {
        "path": "quality/qualifications/australian-pbs-utilisation-landed-lineage-20261019.json",
        "sha256": "758db3eb852dafae15da6979d4ae6638f245a94e23232d901d3d9dde4322bd58",
        "rights_state": "unknown",
        "reviewer_status": "unreviewed",
    },
    "rights_attestation": {
        "path": "quality/qualifications/australian-pbs-utilisation-six-object-rights-attestation-20261020.json",
        "sha256": "903c42f29ac7518a8be0525f0c1488b78d21f25ad051c9d108bfb2b3736b2968",
        "state": "maintainer_attested_exact_public_release_permission_for_pbs_utilisation_raw_object",
        "permission_instrument_independently_reviewed": False,
    },
    "role_readiness": {
        "path": "quality/qualifications/australian-m112-pbs-utilisation-role-readiness-20261028.json",
        "sha256": "fb680271214008a850416151529a51933b16cb8b1521ec156c360872cd46d4b2",
        "sensitivity": "non_sensitive_source_profile_assessment",
        "exhaustive_personal_data_clearance": False,
        "personal_data_absence_proven": False,
    },
    "replay_manifest": {
        "path": f"{RECEIPT_BASE}/replay-manifest.json",
        "sha256": "181877c25380042f277ae7e652b178cea50f3b59f0fcb82f4b38923ee380a770",
    },
    "raw_admission_cohort": {
        "path": "quality/qualifications/australian-m112-pbs-v4-admission-readers-20261028/manifest.json",
        "sha256": "fb09cd52d1d61796af070cbfc29bfe66683206fc5195b9f7c8b6e084e1a7cd0e",
    },
    "effective_raw_delivery": {
        "path": "quality/qualifications/australian-m112-effective-raw-delivery-reconciliation-20261115-august.json",
        "sha256": "759984c23dd557a536463ad4cacee5c2ac6360cd2f74bf619fa03dc3712f351a",
    },
}
QUALIFICATION = {
    "evidence_as_of": "2026-11-15",
    "byte_status": "upstream_hosted_replay_passed",
    "semantic_status": "not_qualified",
    "raw_v4_admission": "six_raw_objects_admitted",
    "raw_reader_canary": "six_passed",
    "modelling_consumer_canary": "not_evidenced",
    "independent_recovery": "not_qualified",
}


def _same(actual, expected):
    """Compare JSON values without treating booleans as integers."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def _expect(actual, expected, label):
    if not _same(actual, expected):
        raise ValueError(f"{label} does not match the frozen evidence")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _json_bytes(payload):
    try:
        return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc


def validate_manifest(manifest: dict[str, typing.Any]) -> None:
    """Reject changes to this exact six-object evidence contract."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "source", "upstream", "qualification", "objects"}:
        raise ValueError("manifest has missing or unknown fields")
    _expect(manifest["schema"], "innovate.pbs-source-manifest.v1", "schema")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {"dataset", "revision"}:
        raise ValueError("source dataset/revision fields are invalid")
    _expect(source["dataset"], DATASET, "dataset")
    _expect(source["revision"], REVISION, "revision")
    upstream = manifest["upstream"]
    if not isinstance(upstream, dict) or set(upstream) != set(UPSTREAM):
        raise ValueError("upstream evidence fields are invalid")
    for key, expected in UPSTREAM.items():
        label = "sensitivity" if key == "role_readiness" else key
        _expect(upstream[key], expected, label)
    _expect(manifest["qualification"], QUALIFICATION, "qualification")
    objects = manifest["objects"]
    if not isinstance(objects, list) or len(objects) != len(OBJECTS):
        raise ValueError("object identity set must contain exactly six objects")
    seen = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"path", "sha256", "byte_count", "source_profile", "receipt"}:
            raise ValueError("object identity fields are invalid")
        path = obj["path"]
        if not isinstance(path, str) or path not in OBJECTS or path in seen:
            raise ValueError("object identity is unknown or duplicated")
        seen.add(path)
        sha256, size, profile, run_id, receipt_sha256 = OBJECTS[path]
        expected = {
            "path": path,
            "sha256": sha256,
            "byte_count": size,
            "source_profile": profile,
            "receipt": {
                "path": f"{RECEIPT_BASE}/{run_id}/gma-m112-v3-replay-receipt-{run_id}-1.json",
                "sha256": receipt_sha256,
                "run_id": run_id,
            },
        }
        _expect(obj, expected, "object identity or receipt")


def validate_replay_receipt(payload: bytes, manifest: dict[str, typing.Any], obj: dict[str, typing.Any]) -> None:
    """Validate the receipt for an object in the frozen manifest."""
    validate_manifest(manifest)
    if not any(_same(obj, candidate) for candidate in manifest["objects"]):
        raise ValueError("receipt object identity is not in the validated manifest")
    receipt = _json_bytes(payload)
    if not isinstance(receipt, dict):
        raise ValueError("receipt is not an object")
    _expect(receipt.get("schema_id"), "global-medicines-atlas.m112-v3-replay-execution-receipt", "receipt schema")
    _expect(receipt.get("schema_version"), 1, "receipt schema version")
    _expect(receipt.get("status"), "passed", "receipt status")
    _expect(receipt.get("candidate_only"), True, "receipt candidate status")
    scope = receipt.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("receipt scope is invalid")
    for key, expected in {**manifest["source"], **{k: obj[k] for k in ("path", "sha256", "byte_count")}}.items():
        _expect(scope.get(key), expected, f"receipt {key}")
    workflow = receipt.get("hosted_workflow")
    if not isinstance(workflow, dict):
        raise ValueError("receipt workflow is invalid")
    _expect(workflow.get("run_id"), obj["receipt"]["run_id"], "receipt run_id")
    _expect(workflow.get("run_attempt"), 1, "receipt run attempt")
    _expect(workflow.get("repository"), UPSTREAM["repository"], "receipt repository")
    _expect(workflow.get("commit"), "8343d511179e71adea825a5f46e7c6cd60b16373", "receipt producer commit")
    _expect(
        workflow.get("run_url"),
        f"https://github.com/{UPSTREAM['repository']}/actions/runs/{obj['receipt']['run_id']}",
        "receipt run URL",
    )
    _expect(
        receipt.get("verification"),
        {
            "anonymous": True,
            "token_free": True,
            "payload_values_read": False,
            "temporary_source_bytes_removed": True,
        },
        "receipt verification",
    )
    _expect(
        receipt.get("boundaries"),
        {
            "m112_consumer_canary_performed": False,
            "v4_admission_performed": False,
            "source_values_or_rows_included": False,
            "public_dataset_mutation": False,
        },
        "receipt boundaries",
    )
    replay = receipt.get("replay")
    registration = replay.get("registration") if isinstance(replay, dict) else None
    if not isinstance(registration, dict):
        raise ValueError("receipt registration is invalid")
    _expect(registration.get("payload_sha256"), obj["sha256"], "receipt registration payload sha256")
    _expect(registration.get("byte_count"), obj["byte_count"], "receipt registration byte_count")
    _expect(registration.get("rights_state"), UPSTREAM["rights_attestation"]["state"], "receipt registration rights")
    _expect(registration.get("payload_values_read"), False, "receipt registration payload values")
    _expect(hashlib.sha256(payload).hexdigest(), obj["receipt"]["sha256"], "receipt sha256")


def validate_payload(stream: typing.BinaryIO, expected_sha256: str, expected_size: int) -> None:
    """Compare bytes against caller-supplied expectations; never acquire them."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid expected sha256")
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("invalid expected size")
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    _expect(size, expected_size, "payload size")
    _expect(digest.hexdigest(), expected_sha256, "payload sha256")


def validate_source_payload(stream: typing.BinaryIO, manifest: dict[str, typing.Any], path: str, revision: str) -> None:
    """Compare supplied bytes to an exact source identity in the frozen manifest."""
    validate_manifest(manifest)
    _expect(revision, manifest["source"]["revision"], "payload revision")
    matches = [obj for obj in manifest["objects"] if obj["path"] == path]
    if len(matches) != 1:
        raise ValueError("payload object identity is not in the validated manifest")
    validate_payload(stream, matches[0]["sha256"], matches[0]["byte_count"])


def _read_under(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("upstream evidence must remain under the supplied root")
    return path.read_bytes()


def validate_upstream_files(manifest: dict[str, typing.Any], root: pathlib.Path) -> None:
    """Check already-present evidence files; this does not read the source data."""
    validate_manifest(manifest)
    evidence = {}
    for name in (
        "historical_rights",
        "rights_attestation",
        "role_readiness",
        "replay_manifest",
        "raw_admission_cohort",
        "effective_raw_delivery",
    ):
        item = manifest["upstream"][name]
        raw = _read_under(root, item["path"])
        _expect(hashlib.sha256(raw).hexdigest(), item["sha256"], f"{name} sha256")
        evidence[name] = raw
    for obj in manifest["objects"]:
        raw = _read_under(root, obj["receipt"]["path"])
        validate_replay_receipt(raw, manifest, obj)
    cohort = _json_bytes(evidence["raw_admission_cohort"])
    if not isinstance(cohort, dict) or not isinstance(cohort.get("objects"), list):
        raise ValueError("raw admission cohort is invalid")
    expected = {obj["path"]: obj for obj in manifest["objects"]}
    for row in cohort["objects"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] not in expected:
            raise ValueError("raw admission cohort object identity is unknown or duplicated")
        obj = expected.pop(row["path"])
        for key, value in {**manifest["source"], **{k: obj[k] for k in ("path", "sha256", "byte_count")}}.items():
            _expect(row.get(key), value, f"raw admission cohort {key}")
        _expect(row.get("raw_v4_admitted"), True, "raw admission")
        _expect(row.get("raw_reader_canary_passed"), True, "raw reader canary")
        _expect(row.get("domain_semantics_qualified"), False, "raw domain semantics")
        _expect(row.get("independent_recovery_qualified"), False, "raw independent recovery")
        run = row.get("native_run")
        if not isinstance(run, dict):
            raise ValueError("raw admission run is invalid")
        _expect(run.get("status"), "completed", "raw admission run status")
        _expect(run.get("conclusion"), "success", "raw admission run conclusion")
        _expect(run.get("head_sha"), "10b104417ccb37da6ec20b8e7151c6486a787eec", "raw admission producer")
        ref = row.get("receipt")
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise ValueError("raw admission receipt reference is invalid")
        raw = _read_under(root, ref["path"])
        _expect(hashlib.sha256(raw).hexdigest(), ref.get("sha256"), "raw admission receipt sha256")
        receipt = _json_bytes(raw)
        admission = receipt.get("admission") if isinstance(receipt, dict) else None
        canary = receipt.get("consumer_canary") if isinstance(receipt, dict) else None
        if not isinstance(admission, dict) or not isinstance(canary, dict):
            raise ValueError("raw admission receipt is invalid")
        for key, value in {**manifest["source"], **{k: obj[k] for k in ("path", "sha256", "byte_count")}}.items():
            _expect(admission.get(key), value, f"raw admission receipt {key}")
        _expect(admission.get("schema_era"), "unqualified-native-schema", "raw admission schema")
        _expect(canary.get("status"), "passed", "raw reader receipt status")
        _expect(
            canary.get("object"),
            {**manifest["source"], "path": obj["path"], "sha256": obj["sha256"], "bytes": obj["byte_count"]},
            "raw reader object identity",
        )
        workflow = canary.get("hosted_workflow")
        if not isinstance(workflow, dict):
            raise ValueError("raw reader workflow is invalid")
        _expect(workflow.get("run_id"), run.get("id"), "raw reader run id")
        _expect(workflow.get("commit"), run.get("head_sha"), "raw reader commit")
        verification = canary.get("verification")
        if not isinstance(verification, dict):
            raise ValueError("raw reader verification is invalid")
        _expect(verification.get("consumer_stream_digest_match"), True, "raw reader digest")
        _expect(verification.get("remote_digest_match"), True, "raw remote digest")
        _expect(verification.get("source_values_or_rows_interpreted"), False, "raw reader source values")
        _expect(verification.get("reader_cache_bytes_after_close"), 0, "raw reader cache cleanup")
    if expected:
        raise ValueError("raw admission cohort is missing an object")


def main() -> None:
    """Run metadata validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=ROOT / "validation/data/pbs-sources.json")
    parser.add_argument("--receipt-root", type=pathlib.Path, help="Existing Atlas checkout containing pinned metadata")
    args = parser.parse_args()
    try:
        manifest = _json_bytes(args.manifest.read_bytes())
        validate_manifest(manifest)
        if args.receipt_root:
            validate_upstream_files(manifest, args.receipt_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"PBS manifest validation failed: {exc}\n")
    detail = " and existing upstream receipt files" if args.receipt_root else " metadata only"
    print(f"Validated frozen PBS manifest{detail}; semantic and modelling qualification remain open.")


if __name__ == "__main__":
    main()
