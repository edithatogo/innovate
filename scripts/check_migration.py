"""Fail closed on feature loss and native migration release claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_EVIDENCE_ERROR = "Release blocked: scientific evidence contract"
INSTALLED_TARGETS = {"linux-x86_64", "macos-arm64", "windows-x86_64"}
SIGNER_WORKFLOW = "edithatogo/innovate/.github/workflows/release-readiness.yml"


SOURCE_PATHS = (
    ".github/workflows",
    ".python-version",
    "bindings",
    "conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/sarima-reference/historical-initialization.json",
    "conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/sarima-reference/stationary-fixed.json",
    "conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/stl-reference-20261107/reference.json",
    "rust",
    "scripts",
    "specs",
    "tests",
    "Cargo.toml",
    "Cargo.lock",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
)


def source_inventory_digest(revision: str) -> str:
    entries = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision, "--", *SOURCE_PATHS], cwd=ROOT)
    if not entries:
        raise ValueError("Release source inventory is empty")
    return hashlib.sha256(entries).hexdigest()


def release_source_digest() -> str:
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal", "--", *SOURCE_PATHS], cwd=ROOT
    ):
        raise ValueError("Release evidence requires clean source inputs")
    return source_inventory_digest("HEAD")


def release_evidence_path(location: object) -> Path:
    if not isinstance(location, str) or not location or "\x00" in location:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires a relative file path")
    path = Path(location)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} references a missing or out-of-scope file")
    return resolved


def release_evidence_json(payload: bytes) -> dict:
    def unique_fields(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} contains duplicate fields")
        return result

    try:
        document = json.loads(payload, object_pairs_hook=unique_fields)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires a JSON object")
    return document


def release_file_ref(ref: object) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires a file and SHA-256")
    if not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires a valid SHA-256")
    return release_evidence_path(ref["path"])


def verify_release_report(ref: object, kind: str, scope: str, source_digest: str) -> dict:
    payload = release_file_ref(ref).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} report digest does not match")
    report = release_evidence_json(payload)
    results = report.get("results")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != kind
        or report.get("source_sha256") != source_digest
        or report.get("status") != "passed"
        or not isinstance(results, dict)
        or scope not in results
    ):
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} report has the wrong source, scope or outcome")
    for counts in results.values():
        if not isinstance(counts, dict) or set(counts) != {"passed", "failed", "skipped", "errors"}:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} needs explicit test outcomes")
        if any(type(count) is not int for count in counts.values()):
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} needs integer test outcomes")
        if counts["passed"] < 1 or any(counts[outcome] != 0 for outcome in ("failed", "skipped", "errors")):
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires executed checks with no unsuccessful outcomes")
    return report


def verify_release_attestation(payload: bytes, bundle: Path, revision: str) -> None:
    # Verify the exact bytes already parsed, without exposing verifier diagnostics to CI logs.
    with tempfile.TemporaryDirectory(prefix="innovate-release-evidence-") as folder:
        subject = Path(folder) / "release-evidence.json"
        subject.write_bytes(payload)
        try:
            subprocess.run(
                [
                    "gh",
                    "attestation",
                    "verify",
                    str(subject),
                    "--bundle",
                    str(bundle),
                    "--repo",
                    "edithatogo/innovate",
                    "--signer-workflow",
                    SIGNER_WORKFLOW,
                    "--signer-digest",
                    revision,
                    "--source-digest",
                    revision,
                    "--predicate-type",
                    "https://slsa.dev/provenance/v1",
                ],
                cwd=ROOT,
                check=True,
                timeout=90,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} lacks a verified release-workflow attestation") from error


def verify_release_evidence(ledger: dict, source_digest: str) -> None:
    details = ledger.get("release_evidence")
    if not isinstance(details, dict) or set(details) != {"manifest", "attestation"}:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} and obligation coverage are not qualified")
    payload = release_evidence_path(details["manifest"]).read_bytes()
    bundle = release_evidence_path(details["attestation"])
    manifest = release_evidence_json(payload)
    revision = manifest.get("source_commit")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "innovate-release-qualification"
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or manifest.get("source_sha256") != source_digest
    ):
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} manifest has the wrong identity or source digest")
    try:
        observed_digest = source_inventory_digest(revision)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} source revision could not be verified") from error
    if observed_digest != source_digest:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} attested source differs from the release inputs")
    obligations = {item["key"]: item for group in ("models", "feature_areas") for item in ledger[group]}
    coverage = manifest.get("obligations")
    if not isinstance(coverage, dict) or coverage.keys() != obligations.keys():
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires every ledger obligation")
    for key, item in obligations.items():
        refs = coverage[key]
        if not isinstance(refs, dict) or set(refs) != {"native", "scientific"}:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} needs native and scientific receipts")
        if refs["scientific"] != {"path": item["evidence"], "sha256": item["evidence_sha256"]}:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} scientific receipt differs from the ledger")
        verify_release_report(refs["native"], "native_execution", key, source_digest)
        verify_release_report(refs["scientific"], "scientific_acceptance", key, source_digest)
    installed = manifest.get("installed_consumers")
    if not isinstance(installed, dict) or installed.keys() != INSTALLED_TARGETS:
        raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires all installed-consumer targets")
    for target, lane in installed.items():
        if not isinstance(lane, dict) or set(lane) != {"artifact", "report"}:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} requires an installed artifact and report")
        artifact = release_file_ref(lane["artifact"])
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report = verify_release_report(lane["report"], "installed_execution", target, source_digest)
        if digest != lane["artifact"]["sha256"] or report.get("artifact_sha256") != digest:
            raise ValueError(f"{RELEASE_EVIDENCE_ERROR} installed artifact digest does not match")
    verify_release_attestation(payload, bundle, revision)


def main() -> int:
    if sys.version_info[:3] != (3, 15, 2):
        raise ValueError("Use the selected repo-local Python 3.15.2 interpreter")
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--release", action="store_true")
    modes.add_argument("--source-digest", action="store_true")
    args = parser.parse_args()
    if args.source_digest:
        print(release_source_digest())
        return 0
    ledger = json.loads((ROOT / "migration-ledger.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if project["build-system"]["build-backend"] != "maturin":
        raise ValueError("Native Maturin build backend is required")
    if project["project"]["dependencies"] != ["polars==2.1.0"]:
        raise ValueError("Polars is the only base Python runtime dependency")
    binding_entries = {
        "python": "innovate/__init__.py",
        "julia": "Project.toml",
        "r": "DESCRIPTION",
    }
    bindings = ROOT / "bindings"
    for language, entry in binding_entries.items():
        if not (bindings / language / entry).is_file():
            raise ValueError(f"Missing required binding: {language}")
    for directory in bindings.iterdir():
        if (
            directory.is_dir()
            and directory.name not in binding_entries
            and any(path.is_file() for path in directory.rglob("*"))
        ):
            raise ValueError(f"Unsupported language binding: {directory.name}")
    expected_models = {
        "bass",
        "complementary_goods",
        "composite",
        "fisher_pry",
        "gompertz",
        "hierarchical",
        "latent_process",
        "lock_in",
        "logistic",
        "lotka_volterra",
        "mixture",
        "multi_product",
        "network_diffusion",
        "norton_bass",
        "policy_hazard",
        "regime_switching",
    }
    if {m["key"] for m in ledger["models"]} != expected_models:
        raise ValueError("Migration ledger lost a required scientific model")
    expected_features = {
        "covariates_and_events",
        "multi_generation_substitution",
        "competitive_ecosystems",
        "hierarchical_mixture_inference",
        "latent_regime_models",
        "network_policy_diffusion",
        "kairos_des_abm",
        "bayesian_inference",
        "bootstrap_uncertainty",
        "batch_fitting",
        "preprocessing",
        "causal_analysis",
        "adopter_classification",
        "data_ingestion_provenance_validation",
        "scenario_registry_execution",
        "explainability_sensitivity_reporting",
        "benchmark_model_cards",
        "plugin_contracts",
        "remote_execution",
        "accelerator_hpc",
        "voiage_arrow_handoff",
        "calibration_identifiability",
    }
    if {f["key"] for f in ledger["feature_areas"]} != expected_features:
        raise ValueError("Migration ledger lost a required current or planned feature area")
    for group in ("models", "feature_areas"):
        items = ledger[group]
        if len(items) != len({item["key"] for item in items}) or any(item["required"] is not True for item in items):
            raise ValueError("Every unique model and feature obligation must remain required")
    for group in ("models", "feature_areas"):
        for item in ledger[group]:
            status = item.get("status")
            if status not in ("pending", "bounded_native", "qualified"):
                raise ValueError(f"Invalid migration status for {item['key']}")
            if status == "qualified" and item.get("remaining"):
                raise ValueError(f"Qualified migration obligation still has remaining work: {item['key']}")
            evidence = item.get("evidence")
            if evidence is None and status == "pending":
                continue
            if not isinstance(evidence, str) or not evidence:
                raise ValueError(f"Missing migration evidence for {item['key']}")
            evidence_path = Path(evidence)
            resolved = (ROOT / evidence_path).resolve()
            if evidence_path.is_absolute() or not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
                raise ValueError(f"Invalid migration evidence for {item['key']}: {evidence}")
            evidence_bytes = resolved.read_bytes()
            if not evidence_bytes.strip():
                raise ValueError(f"Empty migration evidence for {item['key']}: {evidence}")
            if status == "qualified" and item.get("evidence_sha256") != hashlib.sha256(evidence_bytes).hexdigest():
                raise ValueError(f"Qualified migration evidence digest is missing or mismatched for {item['key']}")

    pending = [
        item["key"]
        for group in ("models", "feature_areas")
        for item in ledger[group]
        if item["required"] and item["status"] != "qualified"
    ]
    release_ready = ledger.get("release_ready")
    if not isinstance(release_ready, bool):
        raise ValueError("release_ready must be a JSON boolean")
    if release_ready and pending:
        raise ValueError("release_ready requires all migration obligations to be qualified")
    if args.release and release_ready:
        source_digest = release_source_digest()
        if ledger.get("qualified_source_sha256") != source_digest:
            raise ValueError("Qualified evidence does not match the release source digest")
        verify_release_evidence(ledger, source_digest)
        print("Native migration and scientific release evidence verified.")
        return 0
    if args.release and not release_ready:
        print("Release blocked: scientific migration and acceptance remain incomplete:", ", ".join(pending))
        return 1
    print(
        f"Native development policy valid; {len(pending)} ledger obligations are not declared qualified. "
        "The scientific evidence contract and obligation coverage remain unqualified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
