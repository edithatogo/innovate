"""Registry records cannot turn local preparation into external acceptance."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TRACK = "conductor/tracks/research_software_registry_readiness_20260721"
REPOSITORY = "https://github.com/edithatogo/innovate"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("registry_readiness", ROOT / "scripts/check_registry_readiness.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packet(tmp_path):
    contents = {
        "CITATION.cff": "cff-version: 1.2.0\nversion: 0.7.0a1\n",
        "LICENSE": "Synthetic test licence, not publication evidence\n",
        "pyproject.toml": '[project]\nversion = "0.7.0a1"\n',
        "migration-ledger.json": '{"release_ready": false}\n',
    }
    for name, content in contents.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    dossier = tmp_path / TRACK / "test-dossier.md"
    dossier.parent.mkdir(parents=True)
    dossier.write_text("Synthetic prerequisite assessment; external gates remain open.\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "parent_issue": 398,
        "assessed_revision": "a" * 40,
        "software_version": "0.7.0a1",
        "input_sha256": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in contents},
        "destinations": [
            {
                "key": key,
                "issue": issue,
                "state": "pending",
                "dossier": f"{TRACK}/test-dossier.md",
                "outstanding": ["Provider decision has not been observed."],
                "identifier": None,
                "evidence": None,
            }
            for key, issue in (("software_heritage", 399), ("rrid", 400), ("joss", 401))
        ],
    }


def record_event(tmp_path, destination, state="verified"):
    """Make explicitly synthetic provider receipts for contract testing only."""
    key = destination["key"]
    identifiers = {
        "software_heritage": "swh:1:rev:" + "b" * 40,
        "rrid": "RRID:SCR_123456",
        "joss": "10.21105/joss.12345",
    }
    urls = {
        "software_heritage": "https://archive.softwareheritage.org/api/1/resolve/" + identifiers[key] + "/",
        "rrid": "https://scicrunch.org/resolver/" + identifiers[key],
        "joss": "https://joss.theoj.org/papers/" + identifiers[key],
    }
    identifier = identifiers[key] if state == "verified" else None
    response = tmp_path / TRACK / "evidence" / f"{key}-synthetic.txt"
    response.parent.mkdir(exist_ok=True)
    response.write_text(f"Synthetic test only: {REPOSITORY} {identifier}\n", encoding="utf-8")
    if key == "software_heritage" and state == "verified":
        response.write_text(
            json.dumps({"namespace": "swh", "scheme_version": 1, "object_type": "revision", "object_id": "b" * 40}),
            encoding="utf-8",
        )
    receipt = {
        "provider": key,
        "repository": REPOSITORY,
        "software_version": "0.7.0a1",
        "event": state,
        "checked_at": "2026-11-14T00:00:00Z",
        "url": urls[key],
        "http_status": 200,
        "response_path": response.relative_to(tmp_path).as_posix(),
        "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        "identifier": identifier,
        "summary": "Synthetic test fixture; no actual provider outcome.",
    }
    if key == "software_heritage" and state == "verified":
        receipt["release"] = {"tag": "v0.7.0a1", "commit": "b" * 40}
    path = tmp_path / TRACK / "evidence" / f"{key}-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    destination.update(
        state=state,
        outstanding=[] if state in {"verified", "ineligible"} else ["Provider acceptance pending."],
        identifier=identifier,
        evidence={
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return path, receipt


def rewrite_receipt(path, receipt, destination):
    path.write_text(json.dumps(receipt), encoding="utf-8")
    destination["evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_pending_packet_is_valid_without_claiming_completion(checker, packet, tmp_path):
    report = checker.validate_packet(tmp_path, packet)
    assert report["record_valid"] is True
    assert report["recorded_complete"] is False
    assert report["online_verification_performed"] is False
    assert report["pending_issues"] == [399, 400, 401]


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "boolean_schema",
        "repository",
        "parent",
        "revision",
        "version",
        "missing_input",
        "changed_input",
        "extra_input",
        "missing_destination",
        "duplicate_destination",
        "wrong_issue",
        "invalid_state",
        "empty_outstanding",
        "empty_dossier",
        "missing_dossier",
        "absolute_dossier",
        "outside_dossier",
        "pending_identifier",
        "pending_receipt",
        "unsupported_completion",
    ],
)
def test_invalid_or_stale_packets_fail_closed(checker, packet, tmp_path, mutation):
    entry = packet["destinations"][0]
    if mutation == "schema":
        packet["schema_version"] = 2
    elif mutation == "boolean_schema":
        packet["schema_version"] = True
    elif mutation == "repository":
        packet["repository"] = "https://github.com/edithatogo/kairos"
    elif mutation == "parent":
        packet["parent_issue"] = 399
    elif mutation == "revision":
        packet["assessed_revision"] = "main"
    elif mutation == "version":
        packet["software_version"] = "0.5.0"
    elif mutation == "missing_input":
        del packet["input_sha256"]["CITATION.cff"]
    elif mutation == "changed_input":
        (tmp_path / "CITATION.cff").write_text("changed\n", encoding="utf-8")
    elif mutation == "extra_input":
        packet["input_sha256"]["../private"] = "f" * 64
    elif mutation == "missing_destination":
        packet["destinations"].pop()
    elif mutation == "duplicate_destination":
        packet["destinations"][1] = entry.copy()
    elif mutation == "wrong_issue":
        entry["issue"] = 400
    elif mutation == "invalid_state":
        entry["state"] = "ready_for_review"
    elif mutation == "empty_outstanding":
        entry["outstanding"] = []
    elif mutation == "empty_dossier":
        (tmp_path / entry["dossier"]).write_text(" \n", encoding="utf-8")
    elif mutation == "missing_dossier":
        entry["dossier"] = "missing.md"
    elif mutation == "absolute_dossier":
        entry["dossier"] = str(tmp_path / entry["dossier"])
    elif mutation == "outside_dossier":
        entry["dossier"] = "../outside.md"
    elif mutation == "pending_identifier":
        entry["identifier"] = "swh:1:rev:" + "a" * 40
    elif mutation == "pending_receipt":
        entry["evidence"] = {"path": entry["dossier"], "sha256": "0" * 64}
    else:
        entry.update(state="verified", outstanding=[])
    with pytest.raises(ValueError):
        checker.validate_packet(tmp_path, packet)


def test_symlink_cannot_import_evidence_outside_repository(checker, packet, tmp_path):
    outside = tmp_path.parent / "unrelated-registry-evidence.md"
    outside.write_text("Unrelated external file", encoding="utf-8")
    link = tmp_path / TRACK / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"This runner cannot create symlinks: {exc}")
    packet["destinations"][0]["dossier"] = link.relative_to(tmp_path).as_posix()
    with pytest.raises(ValueError, match="repository"):
        checker.validate_packet(tmp_path, packet)


@pytest.mark.parametrize("state", ["submitted", "verified", "ineligible"])
def test_provider_receipts_record_outcomes_without_online_claims(checker, packet, tmp_path, state):
    for entry in packet["destinations"]:
        record_event(tmp_path, entry, state)
    report = checker.validate_packet(tmp_path, packet)
    assert report["online_verification_performed"] is False
    assert report["recorded_complete"] is (state != "submitted")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "changed_receipt",
        "wrong_provider",
        "wrong_repository",
        "wrong_version",
        "wrong_event",
        "naive_timestamp",
        "invalid_timestamp",
        "http_url",
        "untrusted_host",
        "credential_url",
        "query_only_identifier",
        "fragment_only_identifier",
        "unrelated_resolver_path",
        "http_error",
        "missing_http_status",
        "negative_resolver_body",
        "wrong_resolver_object",
        "missing_response",
        "changed_response",
        "empty_summary",
        "wrong_identifier",
        "placeholder_identifier",
        "identifier_not_in_response",
        "missing_release",
        "wrong_release_commit",
        "unresolved_blocker",
    ],
)
def test_completion_needs_matching_provider_evidence(checker, packet, tmp_path, mutation):
    entry = packet["destinations"][0]
    path, receipt = record_event(tmp_path, entry)
    if mutation == "missing_receipt":
        path.unlink()
    elif mutation == "changed_receipt":
        path.write_text("{}", encoding="utf-8")
    elif mutation == "wrong_provider":
        receipt["provider"] = "rrid"
    elif mutation == "wrong_repository":
        receipt["repository"] = "https://github.com/edithatogo/kairos"
    elif mutation == "wrong_version":
        receipt["software_version"] = "0.5.0"
    elif mutation == "wrong_event":
        receipt["event"] = "submitted"
    elif mutation == "naive_timestamp":
        receipt["checked_at"] = "2026-11-14T00:00:00"
    elif mutation == "invalid_timestamp":
        receipt["checked_at"] = "not a date"
    elif mutation == "http_url":
        receipt["url"] = receipt["url"].replace("https:", "http:")
    elif mutation == "untrusted_host":
        receipt["url"] = "https://archive.softwareheritage.org.invalid/"
    elif mutation == "credential_url":
        receipt["url"] = receipt["url"].replace("https://", "https://user:password@")
    elif mutation == "query_only_identifier":
        receipt["url"] = "https://archive.softwareheritage.org/?q=" + entry["identifier"]
    elif mutation == "fragment_only_identifier":
        receipt["url"] = "https://archive.softwareheritage.org/#" + entry["identifier"]
    elif mutation == "unrelated_resolver_path":
        receipt["url"] = "https://archive.softwareheritage.org/search/" + entry["identifier"]
    elif mutation == "http_error":
        receipt["http_status"] = 404
    elif mutation == "missing_http_status":
        del receipt["http_status"]
    elif mutation in {"negative_resolver_body", "wrong_resolver_object"}:
        payload = {"exception": "NotFoundExc", "reason": "No object found: " + entry["identifier"]}
        if mutation == "wrong_resolver_object":
            payload = {"namespace": "swh", "scheme_version": 1, "object_type": "revision", "object_id": "c" * 40}
        response = tmp_path / receipt["response_path"]
        response.write_text(json.dumps(payload), encoding="utf-8")
        receipt["response_sha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
    elif mutation == "missing_response":
        receipt["response_path"] = f"{TRACK}/evidence/absent.json"
    elif mutation == "changed_response":
        (tmp_path / receipt["response_path"]).write_text("tampered", encoding="utf-8")
    elif mutation == "empty_summary":
        receipt["summary"] = " "
    elif mutation == "wrong_identifier":
        receipt["identifier"] = "swh:1:rev:" + "c" * 40
    elif mutation == "placeholder_identifier":
        receipt["identifier"] = entry["identifier"] = "swh:1:rev:" + "0" * 40
    elif mutation == "identifier_not_in_response":
        response = tmp_path / receipt["response_path"]
        response.write_text("provider request accepted, no archive yet", encoding="utf-8")
        receipt["response_sha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
    elif mutation == "missing_release":
        del receipt["release"]
    elif mutation == "wrong_release_commit":
        receipt["release"]["commit"] = "c" * 40
    else:
        entry["outstanding"] = ["Still missing acceptance"]
    if mutation not in {"missing_receipt", "changed_receipt"}:
        rewrite_receipt(path, receipt, entry)
    with pytest.raises(ValueError):
        checker.validate_packet(tmp_path, packet)


def test_cli_distinguishes_valid_pending_complete_and_invalid(checker, packet, tmp_path, capsys):
    manifest = tmp_path / "readiness.json"
    manifest.write_text(json.dumps(packet), encoding="utf-8")
    args = ["--root", str(tmp_path), "--manifest", "readiness.json", "--json"]
    assert checker.main(args) == 0
    assert json.loads(capsys.readouterr().out)["record_valid"] is True
    assert checker.main([*args, "--require-complete"]) == 1
    capsys.readouterr()
    for entry in packet["destinations"]:
        record_event(tmp_path, entry)
    manifest.write_text(json.dumps(packet), encoding="utf-8")
    assert checker.main([*args, "--require-complete"]) == 0
    capsys.readouterr()
    manifest.write_text("{}", encoding="utf-8")
    assert checker.main(args) == 2
    assert json.loads(capsys.readouterr().out)["record_valid"] is False


def test_real_packet_is_checked_by_the_active_ci_suite():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_registry_readiness.py"), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["online_verification_performed"] is False


def test_duplicate_json_fields_are_rejected(checker, tmp_path, capsys):
    (tmp_path / "duplicate.json").write_text('{"schema_version": 2, "schema_version": 1}', encoding="utf-8")
    assert checker.main(["--root", str(tmp_path), "--manifest", "duplicate.json", "--json"]) == 2
    assert "Duplicate JSON key" in json.loads(capsys.readouterr().out)["error"]
