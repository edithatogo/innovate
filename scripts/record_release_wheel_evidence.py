"""Record observed smoke tests against the exact installed dry-run wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from scripts.check_migration import INSTALLED_TARGETS, ROOT, release_source_digest
from scripts.qualify_scenario_arrow import validate_native_loader, verify_wheel

ORACLE_MODULE = "test_oracle_manifest"
ORACLE_CASES = {
    f"test_installed_native_predictions_match_independent_references[{model}]"
    for model in ("bass", "logistic", "gompertz", "fisher_pry", "norton_bass", "multi_product")
}
SMOKE_MODULES = {
    "test_python_consumer",
    "test_competition",
    "test_multi_generation",
    "test_paired_outcomes",
    ORACLE_MODULE,
}
LIVE_OECD = ("test_python_consumer", "test_live_oecd_fetch_returns_polars_observations")


def smoke_results(payload: bytes) -> tuple[dict[str, int], list[str]]:
    """Count each reported pass once and preserve the existing live-endpoint skip."""
    passed: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    skipped: list[str] = []
    report = ElementTree.fromstring(payload)
    for case in report.iter("testcase"):
        module = case.get("classname", "").rsplit(".", 1)[-1]
        name = case.get("name", "")
        identity = (module, name)
        if module not in SMOKE_MODULES or not name or identity in seen:
            raise ValueError("Unexpected or repeated installed-wheel case")
        seen.add(identity)
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError("An installed-wheel case failed")
        if case.find("skipped") is not None:
            if identity != LIVE_OECD:
                raise ValueError("An installed-wheel case was unexpectedly skipped")
            skipped.append(".".join(LIVE_OECD))
        else:
            passed[module] += 1
    if LIVE_OECD not in seen or passed.keys() != SMOKE_MODULES:
        raise ValueError("The installed-wheel smoke suite is incomplete")
    if not all((ORACLE_MODULE, name) in seen for name in ORACLE_CASES):
        raise ValueError("The installed-wheel analytic model suite is incomplete")
    return dict(passed), skipped


def record(environment: Path, wheel: Path, report: Path, output: Path) -> None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    target = f"{'macos' if system == 'darwin' else system}-{'x86_64' if machine == 'amd64' else machine}"
    if sys.version_info[:3] != (3, 15, 2) or target not in INSTALLED_TARGETS:
        raise ValueError("The installed-wheel record requires Python 3.15.2 on a supported release host")
    import innovate
    from innovate import _native

    imported = [Path(module.__file__).resolve(strict=True) for module in (innovate, _native)]
    if not all(path.is_relative_to(environment.resolve(strict=True)) for path in imported):
        raise ValueError("Innovate must run from the isolated wheel environment")
    validate_native_loader(_native.__loader__)
    artifact = verify_wheel(wheel, imported[0].parent, imported)
    source_digest = release_source_digest()
    if (wheel.parent / "SOURCE_SHA256").read_text(encoding="utf-8").strip() != source_digest:
        raise ValueError("The built wheel names different source inputs")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The observed source revision is invalid")
    payload = report.read_bytes()
    passed, skipped = smoke_results(payload)
    evidence = {
        "schema_version": 1,
        "kind": "bounded_installed_execution",
        "status": "passed",
        "scope": "Installed Python wheel smoke suite",
        "source_commit": revision,
        "source_sha256": source_digest,
        "target": target,
        "python": platform.python_version(),
        "artifact_sha256": artifact["sha256"],
        "verified_package_members": artifact["verified_package_members"],
        "pytest_report_sha256": hashlib.sha256(payload).hexdigest(),
        "passed_by_module": passed,
        "skipped": skipped,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("environment", "wheel", "report", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    record(args.environment, args.wheel, args.report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
