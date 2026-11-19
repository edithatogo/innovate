"""Ordinary policy regressions also used by the bounded local mutation trial.

Load the policy through Python's normal import machinery so mutation tools can
instrument its source. Every ledger and evidence file used here is synthetic.
"""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).parents[2]
EVIDENCE = b"Synthetic evidence for the migration policy contract.\n"
DIGEST = hashlib.sha256(EVIDENCE).hexdigest()
SOURCE_DIGEST = "a" * 64


@pytest.fixture
def policy_tree(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO))
    policy = importlib.import_module("scripts.check_migration")
    monkeypatch.setattr(policy, "ROOT", tmp_path)
    monkeypatch.setattr(policy, "sys", SimpleNamespace(version_info=sys.version_info))
    monkeypatch.setattr(sys, "argv", ["check_migration.py"])
    contract = json.loads((REPO / "migration-ledger.json").read_text())
    ledger = {
        "release_ready": False,
        **{
            group: [{"key": entry["key"], "required": True, "status": "pending"} for entry in contract[group]]
            for group in ("models", "feature_areas")
        },
    }
    for entry in ("python/innovate/__init__.py", "julia/Project.toml", "r/DESCRIPTION"):
        path = tmp_path / "bindings" / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["polars==2.1.0"]\n[build-system]\nbuild-backend = "maturin"\n'
    )
    (tmp_path / "evidence.txt").write_bytes(EVIDENCE)

    def write():
        (tmp_path / "migration-ledger.json").write_text(json.dumps(ledger))

    write()
    return policy, ledger, write


def qualify(ledger):
    for group in ("models", "feature_areas"):
        for item in ledger[group]:
            item.update(status="qualified", evidence="evidence.txt", evidence_sha256=DIGEST)
    ledger.update(release_ready=True, qualified_source_sha256=SOURCE_DIGEST)


def test_pending_development_is_valid_but_release_is_blocked(policy_tree, monkeypatch, capsys):
    policy, ledger, _ = policy_tree
    pending = len(ledger["models"]) + len(ledger["feature_areas"])
    assert policy.main() == 0
    assert f"{pending} ledger obligations are not declared qualified" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--release"])
    assert policy.main() == 1
    message = capsys.readouterr().out
    assert message.startswith("Release blocked:")
    for group in ("models", "feature_areas"):
        assert all(item["key"] in message for item in ledger[group])


def test_qualified_release_requires_matching_source_and_scientific_contract(policy_tree, monkeypatch):
    policy, ledger, write = policy_tree
    qualify(ledger)
    write()
    monkeypatch.setattr(policy, "release_source_digest", lambda: SOURCE_DIGEST)
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--release"])
    with pytest.raises(ValueError, match="scientific evidence contract"):
        policy.main()
    monkeypatch.setattr(policy, "release_source_digest", lambda: "b" * 64)
    with pytest.raises(ValueError, match="does not match the release source digest"):
        policy.main()


@pytest.fixture
def attestation_bundle(policy_tree, monkeypatch):
    policy, ledger, write = policy_tree
    source_commit = "c" * 40
    release_attestations = []

    def evidence(name, value):
        payload = json.dumps(value).encode()
        (policy.ROOT / name).write_bytes(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    def report(name, kind, scopes):
        return evidence(
            name,
            {
                "schema_version": 1,
                "kind": kind,
                "source_sha256": SOURCE_DIGEST,
                "status": "passed",
                "results": {scope: {"passed": 1, "failed": 0, "skipped": 0, "errors": 0} for scope in scopes},
            },
        )

    obligation_items = [item for group in ("models", "feature_areas") for item in ledger[group]]
    ob_keys = [item["key"] for item in obligation_items]
    native = report("native.json", "native_execution", ob_keys)
    scientific = report("science.json", "scientific_acceptance", ob_keys)
    for item in obligation_items:
        item.update(status="qualified", evidence=scientific["path"], evidence_sha256=scientific["sha256"])
    targets = ("linux-x86_64", "macos-arm64", "windows-x86_64")
    installed = {}
    for target in targets:
        artifact = evidence(f"{target}.artifact", {"target": target})
        receipt = report(f"{target}.report.json", "installed_execution", [target])
        data = json.loads((policy.ROOT / receipt["path"]).read_bytes())
        data["artifact_sha256"] = artifact["sha256"]
        receipt = evidence(receipt["path"], data)
        installed[target] = {"report": receipt, "artifact": artifact}
    manifest = {
        "schema_version": 1,
        "kind": "innovate-release-qualification",
        "source_commit": source_commit,
        "source_sha256": SOURCE_DIGEST,
        "obligations": {key: {"native": native, "scientific": scientific} for key in ob_keys},
        "installed_consumers": installed,
    }
    (policy.ROOT / "attestation.jsonl").write_text("synthetic verifier input", encoding="utf-8")

    def save():
        release = evidence("release.json", manifest)
        ledger.update(
            release_ready=True,
            qualified_source_sha256=SOURCE_DIGEST,
            release_evidence={"manifest": release["path"], "attestation": "attestation.jsonl"},
        )
        write()

    def verify_receipt(payload, bundle, revision):
        release_attestations.append((payload, bundle.name, revision))

    monkeypatch.setattr(policy, "release_source_digest", lambda: SOURCE_DIGEST)
    monkeypatch.setattr(policy, "source_inventory_digest", lambda revision: SOURCE_DIGEST)
    monkeypatch.setattr(policy, "verify_release_attestation", verify_receipt)
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--release"])
    return policy, manifest, save, release_attestations, source_commit


def test_release_evidence_binds_all_obligations_and_artifacts(attestation_bundle, capsys):
    policy, _, save, verified, source_commit = attestation_bundle
    save()
    assert policy.main() == 0
    assert verified == [((policy.ROOT / "release.json").read_bytes(), "attestation.jsonl", source_commit)]
    assert "scientific release evidence verified" in capsys.readouterr().out


@pytest.mark.parametrize(
    "changed",
    [
        "omitted_obligation",
        "omitted_platform",
        "mismatched_source",
        "changed_artifact",
        "wrong_receipt_role",
        "skipped_installed_case",
    ],
)
def test_release_evidence_rejects_incomplete_or_mismatched_reports(attestation_bundle, changed):
    policy, manifest, save, _, _ = attestation_bundle
    if changed == "omitted_obligation":
        del manifest["obligations"]["bass"]
    elif changed == "omitted_platform":
        del manifest["installed_consumers"]["macos-arm64"]
    elif changed == "mismatched_source":
        manifest["source_sha256"] = "b" * 64
    elif changed == "changed_artifact":
        (policy.ROOT / "linux-x86_64.artifact").write_bytes(b"changed bytes")
    elif changed == "skipped_installed_case":
        ref = manifest["installed_consumers"]["linux-x86_64"]["report"]
        report_path = policy.ROOT / ref["path"]
        report = json.loads(report_path.read_bytes())
        report["results"]["linux-x86_64"]["skipped"] = 1
        payload = json.dumps(report).encode()
        report_path.write_bytes(payload)
        ref["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        manifest["obligations"]["bass"]["native"] = manifest["obligations"]["bass"]["scientific"]
    save()
    with pytest.raises(ValueError, match="scientific evidence contract"):
        policy.main()


def test_release_bundle_verification_pins_subject_and_workflow(policy_tree, monkeypatch):
    policy, _, _ = policy_tree
    bundle = policy.ROOT / "candidate-attestation.jsonl"

    def reject_verification(command, **options):
        assert command[:3] == ["gh", "attestation", "verify"]
        assert Path(command[3]).read_bytes() == b"exact manifest"
        assert command[4:] == [
            "--bundle",
            str(bundle),
            "--repo",
            "edithatogo/innovate",
            "--signer-workflow",
            "edithatogo/innovate/.github/workflows/release-readiness.yml",
            "--signer-digest",
            "c" * 40,
            "--source-digest",
            "c" * 40,
            "--predicate-type",
            "https://slsa.dev/provenance/v1",
        ]
        assert options["check"] is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(policy.subprocess, "run", reject_verification)
    with pytest.raises(ValueError, match="lacks a verified release-workflow attestation"):
        policy.verify_release_attestation(b"exact manifest", bundle, "c" * 40)


def test_qualified_but_unapproved_release_remains_blocked(policy_tree, monkeypatch):
    policy, ledger, write = policy_tree
    qualify(ledger)
    ledger["release_ready"] = False
    write()
    assert policy.main() == 0
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--release"])
    assert policy.main() == 1


def test_development_output_does_not_claim_qualified_scientific_evidence(policy_tree, capsys):
    policy, ledger, write = policy_tree
    qualify(ledger)
    write()
    assert policy.main() == 0
    output = capsys.readouterr().out
    assert "0 ledger obligations are not declared qualified" in output
    assert "scientific evidence contract and obligation coverage remain unqualified" in output


def test_source_digest_mode_does_not_require_a_ledger(policy_tree, monkeypatch, capsys):
    policy, _, _ = policy_tree
    (policy.ROOT / "migration-ledger.json").unlink()
    monkeypatch.setattr(policy, "release_source_digest", lambda: SOURCE_DIGEST)
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--source-digest"])
    assert policy.main() == 0
    assert capsys.readouterr().out == SOURCE_DIGEST + "\n"


def test_release_and_source_digest_modes_are_mutually_exclusive(policy_tree, monkeypatch):
    policy, _, _ = policy_tree
    monkeypatch.setattr(policy, "release_source_digest", lambda: SOURCE_DIGEST)
    monkeypatch.setattr(sys, "argv", ["check_migration.py", "--release", "--source-digest"])
    with pytest.raises(SystemExit) as error:
        policy.main()
    assert error.value.code == 2


def test_source_inventory_must_be_nonempty_and_clean(policy_tree, monkeypatch):
    policy, _, _ = policy_tree
    source_root = policy.ROOT / "synthetic-git-repo"
    source_root.mkdir()
    monkeypatch.setattr(policy, "ROOT", source_root)

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Synthetic policy test",
                "-c",
                "user.email=policy@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=source_root,
        )

    git("init", "--quiet")
    git("commit", "--quiet", "--allow-empty", "-m", "Synthetic empty inventory")
    with pytest.raises(ValueError, match="Release source inventory is empty"):
        policy.release_source_digest()
    source = source_root / "Cargo.toml"
    source.write_text("synthetic source\n")
    git("add", "Cargo.toml")
    git("commit", "--quiet", "-m", "Synthetic source inventory")
    expected = hashlib.sha256(git("ls-tree", "-r", "-z", "HEAD", "--", "Cargo.toml")).hexdigest()
    assert policy.release_source_digest() == expected
    source.write_text("changed synthetic source\n")
    with pytest.raises(ValueError, match="Release evidence requires clean source inputs"):
        policy.release_source_digest()


@pytest.mark.parametrize("version", [(3, 15, 1), (3, 15, 3), (3, 14, 2), (3, 16, 2)])
def test_other_python_versions_are_rejected(policy_tree, version):
    policy, _, _ = policy_tree
    policy.sys.version_info = version
    with pytest.raises(ValueError, match="selected repo-local Python 3.15.2"):
        policy.main()


@pytest.mark.parametrize(
    ("backend", "dependencies", "message"),
    [
        ("setuptools.build_meta", ["polars==2.1.0"], "Maturin build backend"),
        ("maturin", [], "only base Python runtime dependency"),
        ("maturin", ["polars>=2.1.0"], "only base Python runtime dependency"),
        ("maturin", ["polars==2.1.0", "numpy"], "only base Python runtime dependency"),
    ],
)
def test_build_contract_must_match(policy_tree, backend, dependencies, message):
    policy, _, _ = policy_tree
    (policy.ROOT / "pyproject.toml").write_text(
        f"[project]\ndependencies = {json.dumps(dependencies)}\n[build-system]\nbuild-backend = {json.dumps(backend)}\n"
    )
    with pytest.raises(ValueError, match=message):
        policy.main()


@pytest.mark.parametrize("binding", ["python/innovate/__init__.py", "julia/Project.toml", "r/DESCRIPTION"])
def test_required_binding_must_be_a_file(policy_tree, binding):
    policy, _, _ = policy_tree
    path = policy.ROOT / "bindings" / binding
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="Missing required binding"):
        policy.main()


def test_empty_extra_binding_directory_is_allowed_but_implemented_one_is_rejected(policy_tree):
    policy, _, _ = policy_tree
    (policy.ROOT / "bindings" / "README.md").write_text("Synthetic binding note")
    extra = policy.ROOT / "bindings" / "java"
    extra.mkdir()
    assert policy.main() == 0
    (extra / "pom.xml").write_text("<synthetic />")
    with pytest.raises(ValueError, match="Unsupported language binding: java"):
        policy.main()


@pytest.mark.parametrize("group", ["models", "feature_areas"])
@pytest.mark.parametrize("change", ["drop", "unknown", "duplicate"])
def test_obligation_inventory_cannot_change(policy_tree, group, change):
    policy, ledger, write = policy_tree
    if change == "drop":
        ledger[group].pop()
    elif change == "unknown":
        ledger[group][0]["key"] = "synthetic_unknown_obligation"
    else:
        ledger[group].append(dict(ledger[group][0]))
    write()
    with pytest.raises(ValueError, match="lost|required"):
        policy.main()


@pytest.mark.parametrize("required", [False, 0, 1, None, "true"])
def test_required_flag_is_exactly_true(policy_tree, required):
    policy, ledger, write = policy_tree
    ledger["models"][0]["required"] = required
    write()
    with pytest.raises(ValueError, match="must remain required"):
        policy.main()


@pytest.mark.parametrize("status", [None, "complete", "", "QUALIFIED"])
def test_unknown_status_is_rejected(policy_tree, status):
    policy, ledger, write = policy_tree
    ledger["models"][0]["status"] = status
    write()
    with pytest.raises(ValueError, match="Invalid migration status"):
        policy.main()


@pytest.mark.parametrize("status", ["pending", "bounded_native", "qualified"])
def test_valid_evidence_is_accepted_for_each_status(policy_tree, status):
    policy, ledger, write = policy_tree
    ledger["models"][0].update(status=status, evidence="evidence.txt", evidence_sha256=DIGEST)
    write()
    assert policy.main() == 0


@pytest.mark.parametrize("status", ["bounded_native", "qualified"])
@pytest.mark.parametrize("evidence", [None, "", 0, False, []])
def test_nonpending_status_requires_named_evidence(policy_tree, status, evidence):
    policy, ledger, write = policy_tree
    ledger["models"][0].update(status=status, evidence=evidence)
    write()
    with pytest.raises(ValueError, match="Missing migration evidence"):
        policy.main()


@pytest.mark.parametrize("kind", ["absolute", "outside", "missing", "directory", "symlink"])
def test_evidence_must_resolve_to_a_file_inside_the_repository(policy_tree, kind):
    policy, ledger, write = policy_tree
    outside = policy.ROOT.parent / "synthetic-outside-evidence.txt"
    outside.write_bytes(EVIDENCE)
    evidence = "evidence.txt"
    if kind == "absolute":
        evidence = str(policy.ROOT / evidence)
    elif kind == "outside":
        evidence = "../synthetic-outside-evidence.txt"
    elif kind == "missing":
        evidence = "missing.txt"
    elif kind == "directory":
        evidence = "bindings"
    else:
        (policy.ROOT / "outside-link.txt").symlink_to(outside)
        evidence = "outside-link.txt"
    ledger["models"][0].update(status="bounded_native", evidence=evidence)
    write()
    with pytest.raises(ValueError, match="Invalid migration evidence"):
        policy.main()


@pytest.mark.parametrize("evidence_bytes", [b"", b" \n\t"])
def test_empty_evidence_cannot_qualify_an_obligation(policy_tree, evidence_bytes):
    policy, ledger, write = policy_tree
    (policy.ROOT / "evidence.txt").write_bytes(evidence_bytes)
    ledger["models"][0].update(status="bounded_native", evidence="evidence.txt")
    write()
    with pytest.raises(ValueError, match="Empty migration evidence"):
        policy.main()


@pytest.mark.parametrize("digest", [None, "", "b" * 64])
def test_qualified_evidence_requires_its_actual_digest(policy_tree, digest):
    policy, ledger, write = policy_tree
    ledger["models"][0].update(status="qualified", evidence="evidence.txt", evidence_sha256=digest)
    write()
    with pytest.raises(ValueError, match="digest is missing or mismatched"):
        policy.main()


@pytest.mark.parametrize("remaining", [None, "", "Scientific parity remains unqualified"])
def test_only_qualified_obligations_prohibit_remaining_work(policy_tree, remaining):
    policy, ledger, write = policy_tree
    item = ledger["models"][0]
    item.update(status="bounded_native", evidence="evidence.txt", remaining=remaining, evidence_sha256=DIGEST)
    write()
    assert policy.main() == 0
    item["status"] = "qualified"
    write()
    if remaining:
        with pytest.raises(ValueError, match="still has remaining work"):
            policy.main()
    else:
        assert policy.main() == 0


@pytest.mark.parametrize("release_ready", [None, 0, 1, "false", []])
def test_release_ready_requires_a_boolean(policy_tree, release_ready):
    policy, ledger, write = policy_tree
    ledger["release_ready"] = release_ready
    write()
    with pytest.raises(ValueError, match="release_ready must be a JSON boolean"):
        policy.main()


@pytest.mark.parametrize("status", ["pending", "bounded_native"])
def test_one_unqualified_obligation_prevents_release_readiness(policy_tree, status):
    policy, ledger, write = policy_tree
    qualify(ledger)
    ledger["feature_areas"][-1]["status"] = status
    write()
    with pytest.raises(ValueError, match="requires all migration obligations to be qualified"):
        policy.main()
