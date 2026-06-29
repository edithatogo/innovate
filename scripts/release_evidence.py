"""Generate coverage and mutation evidence artifacts for release readiness."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Thresholds
COVERAGE_THRESHOLD_LINE_RATE = 0.80
MUTATION_SCORE_THRESHOLD = 0.70

EVIDENCE_ROOT = Path("docs/source/_static/release_readiness/evidence")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_coverage_evidence(
    *,
    line_rate: float,
    branch_rate: float,
    lines_covered: int,
    lines_valid: int,
    branches_covered: int,
    branches_valid: int,
    summary: str = "",
) -> dict[str, Any]:
    """Build a coverage evidence dict with pass/fail based on line_rate threshold."""
    passed = line_rate >= COVERAGE_THRESHOLD_LINE_RATE
    return {
        "evidence_id": "coverage",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "summary": summary,
        "line_rate": line_rate,
        "branch_rate": branch_rate,
        "lines_covered": lines_covered,
        "lines_valid": lines_valid,
        "branches_covered": branches_covered,
        "branches_valid": branches_valid,
        "python_version": "3.14",
        "requires_secrets": False,
        "generated_at": _now_iso(),
        "command": "uv run nox -s coverage",
    }


def build_mutation_evidence(
    *,
    score: float,
    mutants_killed: int,
    mutants_total: int,
    summary: str = "",
) -> dict[str, Any]:
    """Build a mutation evidence dict with pass/fail based on score threshold."""
    passed = score >= MUTATION_SCORE_THRESHOLD
    return {
        "evidence_id": "mutation_sampling",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "summary": summary
        if not passed
        else f"Mutation score {score:.1%} meets threshold {MUTATION_SCORE_THRESHOLD:.0%}",
        "mutation_score": score,
        "mutants_killed": mutants_killed,
        "mutants_total": mutants_total,
        "python_version": "3.14",
        "requires_secrets": False,
        "generated_at": _now_iso(),
        "command": "uv run nox -s mutation",
    }


def _parse_mutmut_results() -> dict[str, Any]:
    """Parse mutmut results from stdout or cached results."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "python", "-m", "mutmut", "results"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout or result.stderr

    # Parse mutmut output to extract score
    # Expected format: "67/100 🎯 67.0%"
    import re

    score_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if score_match:
        killed = int(score_match.group(1))
        total = int(score_match.group(2))
        score = killed / total if total > 0 else 0.0
    else:
        # Try percentage match: "67.0%"
        pct_match = re.search(r"([\d.]+)\s*%", text)
        if pct_match:
            score = float(pct_match.group(1)) / 100.0
            killed = 0
            total = 0
        else:
            score = 0.0
            killed = 0
            total = 0

    return {"score": score, "killed": killed, "total": total}


def write_coverage_evidence(
    *,
    line_rate: float,
    branch_rate: float,
    lines_covered: int,
    lines_valid: int,
    branches_covered: int,
    branches_valid: int,
    summary: str = "",
    output_dir: Path = EVIDENCE_ROOT,
) -> Path:
    """Build and write coverage evidence to the release readiness evidence directory."""
    evidence = build_coverage_evidence(
        line_rate=line_rate,
        branch_rate=branch_rate,
        lines_covered=lines_covered,
        lines_valid=lines_valid,
        branches_covered=branches_covered,
        branches_valid=branches_valid,
        summary=summary,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "coverage.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_mutation_evidence(
    *,
    score: float,
    mutants_killed: int,
    mutants_total: int,
    summary: str = "",
    output_dir: Path = EVIDENCE_ROOT,
) -> Path:
    """Build and write mutation evidence to the release readiness evidence directory."""
    evidence = build_mutation_evidence(
        score=score,
        mutants_killed=mutants_killed,
        mutants_total=mutants_total,
        summary=summary,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mutation-sampling.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main_cli() -> None:
    """CLI entrypoint for generating mutation evidence from mutmut output."""
    parsed = _parse_mutmut_results()
    write_mutation_evidence(
        score=parsed["score"],
        mutants_killed=parsed["killed"],
        mutants_total=parsed["total"],
        summary=f"Mutants killed: {parsed['killed']}/{parsed['total']}",
    )
    print(f"Mutation evidence written: score={parsed['score']:.1%}")


if __name__ == "__main__":
    main_cli()
