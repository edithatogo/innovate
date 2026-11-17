"""Offline adversarial checks for the six-object raw admission chain."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pbs_upstream", ROOT / "scripts/validate_pbs_source_manifest.py")
assert SPEC is not None and SPEC.loader is not None
pbs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pbs)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write(root, path, raw):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def synthetic_chain(tmp_path, monkeypatch):
    """Build an offline metadata chain; it contains no archived source bytes."""
    manifest = json.loads((ROOT / "validation/data/pbs-sources.json").read_text())
    expected_upstream = copy.deepcopy(pbs.UPSTREAM)
    expected_objects = copy.deepcopy(pbs.OBJECTS)
    template = json.loads((ROOT / "tests/validation/fixtures/pbs-replay-40677682866.json").read_text())
    cohort = {"objects": []}
    for offset, obj in enumerate(manifest["objects"]):
        identity = {**manifest["source"], **{key: obj[key] for key in ("path", "sha256", "byte_count")}}
        replay = copy.deepcopy(template)
        replay["scope"].update(identity)
        replay["hosted_workflow"].update(
            run_id=obj["receipt"]["run_id"],
            run_url=f"https://github.com/{expected_upstream['repository']}/actions/runs/{obj['receipt']['run_id']}",
        )
        replay["replay"]["registration"].update(payload_sha256=obj["sha256"], byte_count=obj["byte_count"])
        replay_digest = _write(tmp_path, obj["receipt"]["path"], _bytes(replay))
        obj["receipt"]["sha256"] = replay_digest
        expected_objects[obj["path"]] = (*expected_objects[obj["path"]][:-1], replay_digest)

        run_id = 2000 + offset
        run_sha = "10b104417ccb37da6ec20b8e7151c6486a787eec"
        raw_receipt = {
            "admission": {**identity, "schema_era": "unqualified-native-schema"},
            "consumer_canary": {
                "status": "passed",
                "object": {
                    **manifest["source"],
                    "path": obj["path"],
                    "sha256": obj["sha256"],
                    "bytes": obj["byte_count"],
                },
                "hosted_workflow": {"run_id": run_id, "commit": run_sha},
                "verification": {
                    "consumer_stream_digest_match": True,
                    "remote_digest_match": True,
                    "source_values_or_rows_interpreted": False,
                    "reader_cache_bytes_after_close": 0,
                },
            },
        }
        raw_path = f"synthetic/raw-reader-{offset}.json"
        raw_digest = _write(tmp_path, raw_path, _bytes(raw_receipt))
        cohort["objects"].append(
            {
                **identity,
                "raw_v4_admitted": True,
                "raw_reader_canary_passed": True,
                "domain_semantics_qualified": False,
                "independent_recovery_qualified": False,
                "native_run": {"id": run_id, "status": "completed", "conclusion": "success", "head_sha": run_sha},
                "receipt": {"path": raw_path, "sha256": raw_digest},
            }
        )

    for name in (
        "historical_rights",
        "rights_attestation",
        "role_readiness",
        "replay_manifest",
        "raw_admission_cohort",
        "effective_raw_delivery",
    ):
        item = expected_upstream[name]
        raw = _bytes(cohort if name == "raw_admission_cohort" else {"synthetic_evidence": name})
        item["sha256"] = _write(tmp_path, item["path"], raw)
    manifest["upstream"] = copy.deepcopy(expected_upstream)
    monkeypatch.setattr(pbs, "UPSTREAM", expected_upstream)
    monkeypatch.setattr(pbs, "OBJECTS", expected_objects)
    return manifest, tmp_path, cohort


def _replace_cohort(manifest, root, cohort):
    evidence = manifest["upstream"]["raw_admission_cohort"]
    digest = _write(root, evidence["path"], _bytes(cohort))
    evidence["sha256"] = digest
    pbs.UPSTREAM["raw_admission_cohort"]["sha256"] = digest


def test_all_synthetic_replay_and_raw_reader_receipts_are_checked(synthetic_chain):
    manifest, root, _ = synthetic_chain
    pbs.validate_upstream_files(manifest, root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "unknown or duplicated"),
        ("missing", "missing an object"),
        ("domain_claim", "raw domain semantics"),
        ("failed_run", "raw admission run status"),
        ("missing_run", "raw admission run is invalid"),
        ("changed_receipt", "raw admission receipt sha256"),
        ("changed_object", "raw admission cohort sha256"),
    ],
)
def test_admission_chain_rejects_unqualified_or_tampered_evidence(synthetic_chain, mutation, message):
    manifest, root, cohort = synthetic_chain
    row = cohort["objects"][0]
    if mutation == "duplicate":
        cohort["objects"][1] = copy.deepcopy(row)
    elif mutation == "missing":
        cohort["objects"].pop()
    elif mutation == "domain_claim":
        row["domain_semantics_qualified"] = True
    elif mutation == "failed_run":
        row["native_run"]["status"] = "failed"
    elif mutation == "missing_run":
        row["native_run"] = None
    elif mutation == "changed_receipt":
        row["receipt"]["sha256"] = "0" * 64
    elif mutation == "changed_object":
        row["sha256"] = "0" * 64
    _replace_cohort(manifest, root, cohort)
    with pytest.raises(ValueError, match=message):
        pbs.validate_upstream_files(manifest, root)
