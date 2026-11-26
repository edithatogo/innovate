"""Run Innovate's exact pricing model against the qualified Paignia archives."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

APP_CRATE = Path("rust/crates/innovate-games")
PROVIDER_URL = "https://github.com/edithatogo/paignia.git"
TOOLCHAIN = "1.99.1"
ADMISSION_HELPER_SHA256 = "5994e46d131b16033f598c5e5bc41c9c664cf64ca3637bf333db6169e13eebd5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "application",
        "provider",
        "handoff",
        "archives",
        "registry",
        "work_dir",
        "admission_helper",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if platform.python_implementation() != "CPython" or sys.version_info[:3] != (3, 15, 2):
        raise ValueError("use the qualified application CPython 3.15.2 environment")
    if sys.platform != "darwin":
        raise ValueError("this run uses the macOS source-isolation policy")
    helper = args.admission_helper.read_bytes()
    if hashlib.sha256(helper).hexdigest() != ADMISSION_HELPER_SHA256:
        raise ValueError("the admission helper differs from the retained reimbursement verifier")
    root = args.application.resolve()
    provider = args.provider.resolve()
    registry = args.registry.resolve()
    cargo_git = (registry.parent / "git").resolve()
    work_dir = args.work_dir.resolve()
    if any(work_dir.is_relative_to(path) for path in (provider, root, cargo_git)):
        raise ValueError("the staged application must be outside all source locations")
    stage = Path(tempfile.mkdtemp(dir=work_dir, prefix="innovate-consumer."))
    print(f"Evidence directory: {stage}", flush=True)

    source = stage / "admission_helper.py"
    source.write_bytes(helper)
    spec = importlib.util.spec_from_file_location("receipt_verifier", source)
    if spec is None or spec.loader is None:
        raise ValueError("provide the pinned admission helper as a Python source file")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.ROOT = root
    verifier.PROVIDER = PROVIDER_URL
    verifier.CRATE = APP_CRATE
    verifier.EXPECTED_TESTS = 5
    log = stage / "execution.log"
    handoff = json.loads(args.handoff.read_text())
    artifacts = verifier.verify_and_unpack(provider, handoff, args.archives.resolve(), stage)
    revision = args.revision
    tree = verifier.application_tree(revision)
    snapshot = stage / "application.tar"
    subprocess.run(
        ["git", "archive", "--output", str(snapshot), revision],
        cwd=verifier.ROOT,
        check=True,
    )
    application = stage / "application"
    with tarfile.open(snapshot) as archive:
        archive.extractall(application, filter="data")
    manifest = application / APP_CRATE / "Cargo.toml"
    lockfile = manifest.with_name("Cargo.lock")
    original_lock_hash = verifier.sha256(lockfile)
    locked = tomllib.loads(lockfile.read_text())
    manifest_text = manifest.read_text()
    original_manifest = tomllib.loads(manifest_text)
    provider_revision = handoff["provider_revision"]
    for name, (folder, version) in artifacts.items():
        expected = {
            "git": PROVIDER_URL,
            "rev": provider_revision,
            "version": f"={version}",
            "default-features": False,
        }
        if original_manifest["dependencies"].get(name) != expected:
            raise ValueError(f"Git pin or feature mismatch for {name}")
        path = json.dumps(os.path.relpath(folder, manifest.parent))
        replacement = f'{name} = {{ path = {path}, version = "={version}", default-features = false }}'
        manifest_text, count = re.subn(
            rf"^{name} = .*$",
            lambda _, line=replacement: line,
            manifest_text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected one declaration for {name}")
    core_path = json.dumps(os.path.relpath(artifacts["paignia-core"][0], manifest.parent))
    manifest.write_text(f"{manifest_text}\n[patch.crates-io]\npaignia-core = {{ path = {core_path} }}\n")
    cargo_home = stage / "cargo-home"
    verifier.stage_registry(registry, cargo_home, locked)
    cargo = Path(
        subprocess.check_output(
            ["rustup", "which", "--toolchain", TOOLCHAIN, "cargo"],
            text=True,
        ).strip()
    ).resolve()
    toolchain = cargo.parent.parent
    sandbox_path = stage / "source-isolation.sb"
    sandbox_path.write_text(verifier.SANDBOX)
    build_tmp = stage / "tmp"
    build_tmp.mkdir()
    sdk = subprocess.check_output(["/usr/bin/xcrun", "--show-sdk-path"], text=True).strip()
    env = {
        "PATH": f"{cargo.parent}:/usr/bin:/bin",
        "SDKROOT": sdk,
        "LC_ALL": "C",
        "CARGO_HOME": str(cargo_home),
        "CARGO_TARGET_DIR": str(stage / "build"),
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TERM_COLOR": "never",
        "RUSTC": str(cargo.with_name("rustc")),
        "TMPDIR": str(build_tmp),
    }
    wrapper = [
        "/usr/bin/sandbox-exec",
        "-D",
        f"STAGE={stage}",
        "-D",
        f"PROVIDER={provider}",
        "-D",
        f"CARGO_GIT={cargo_git}",
        "-D",
        f"TOOLCHAIN={toolchain}",
        "-f",
        str(sandbox_path),
    ]
    aliases = (
        (stage, "$STAGE"),
        (provider, "$PROVIDER"),
        (cargo_git, "$CARGO_GIT"),
        (toolchain, "$TOOLCHAIN"),
        (verifier.ROOT, "$APPLICATION_CHECKOUT"),
    )
    commands = []

    def retain(text):
        for path, label in aliases:
            text = text.replace(str(path), label)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        return text

    def run(command, *, denied=False):
        commands.append(retain("$ " + shlex.join(wrapper + command))[2:])
        result = subprocess.run(
            wrapper + command,
            cwd=application,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        retain(result.stdout + result.stderr + f"exit={result.returncode}")
        if denied:
            if result.returncode != 1 or "Operation not permitted" not in result.stderr:
                raise ValueError("sandbox did not deny the selected source read")
        else:
            result.check_returncode()
        return result.stdout

    retain(
        f"{platform.platform()} | {sys.version}\n"
        f"Application revision {revision}\nApplication tree {tree}\n"
        f"Application archive SHA-256 {verifier.sha256(snapshot)}\n"
        f"Original lock SHA-256 {original_lock_hash}\n"
        f"Driver SHA-256 {verifier.sha256(Path(__file__))}\n"
        "Admission helpers reused from new-drug-reimbursement-game b669bd00.\n"
        f"Admission helper SHA-256 {verifier.sha256(source)}\n"
        f"Provider handoff SHA-256 {verifier.sha256(args.handoff)}\n"
        f"{json.dumps(handoff, indent=2)}\n"
        "The same sandbox and environment wrap all commands. Local paths are aliased.\n"
        f"{json.dumps(env, indent=2)}\n{verifier.SANDBOX}"
    )
    for probe in (
        provider / "README.md",
        verifier.ROOT / "Cargo.toml",
        *sorted((cargo_git / "checkouts").glob("paignia-*/*/Cargo.toml")),
    ):
        if not probe.is_file():
            raise ValueError("expected a real source file for the read-denial probe")
        run(["/bin/cat", str(probe)], denied=True)
    run(["/bin/cat", str(manifest)])
    run([str(cargo), "--version"])
    if run([env["RUSTC"], "--version"]).split()[1] != TOOLCHAIN:
        raise ValueError("unexpected application compiler")
    metadata_args = [
        str(cargo),
        "metadata",
        "--offline",
        "--format-version",
        "1",
        "--manifest-path",
        str(manifest),
    ]
    run(metadata_args)
    metadata_text = run([*metadata_args, "--locked"])
    metadata = json.loads(metadata_text)
    verifier.verify_resolution(
        metadata,
        locked,
        tomllib.loads(lockfile.read_text()),
        artifacts,
        provider_revision,
    )
    packages = {item["name"] for item in metadata["packages"]}
    if any(name.startswith("kairo-ecs-") for name in packages) or packages & {
        "innovate-core",
        "innovate-kairos",
        "paignia-kairos",
    }:
        raise ValueError("static consumer retains a simulation dependency")
    retain(f"Resolved lock SHA-256 {verifier.sha256(lockfile)}\n{lockfile.read_text()}")
    retain("Resolved normal and build closure: " + ", ".join(sorted(packages)))
    (stage / "resolved-graph.json").write_text(metadata_text)
    outcome = run(
        [
            str(cargo),
            "test",
            "--offline",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--test",
            "price_commitment",
        ]
    )
    passed = verifier.passing_tests(outcome)
    receipt = {
        "schema_version": 1,
        "scope": "real-application-artifact-consumer",
        "case_kind": "application-model",
        "repository": "https://github.com/edithatogo/innovate",
        "revision": revision,
        "case_name": "Innovate sequential price commitment with structural demand",
        "expected_result_basis": "Independent rational multinomial-logit shares with"
        " an outside option. See the T17.10 static price commitment section in"
        " conductor/tracks/kairos_runtime_boundary_20261107/strategic-information-integration.md"
        " at the application revision.",
        "provider_revision": provider_revision,
        "semantic_profile": verifier.PROFILE,
        "profile_sha256": handoff["profile_contract"]["sha256"],
        "artifacts": handoff["artifacts"],
        "result": "passed",
        "tests_passed": passed,
        "tests_failed": 0,
        "tests_skipped": 0,
        "commands": commands,
        "execution_evidence": {
            "path": "docs/validation/receipts/t17-innovate-price-commitment.log",
            "sha256": verifier.sha256(log),
        },
    }
    receipt_path = stage / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{passed} artifact tests passed at {revision}")
    print(f"{hashlib.sha256(receipt_path.read_bytes()).hexdigest()}  {receipt_path}")


if __name__ == "__main__":
    main()
