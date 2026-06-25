"""Generate non-mutating dependency freshness dashboards across ecosystems."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs/source/_static/dependency_dashboard"


@dataclass(frozen=True)
class Check:
    """A dependency freshness check command."""

    ecosystem: str
    check_id: str
    command: list[str]
    cwd: Path
    required_tools: tuple[str, ...]
    allow_nonzero: bool = True


CHECKS: tuple[Check, ...] = (
    Check(
        ecosystem="python",
        check_id="uv-tree-outdated",
        command=["uv", "tree", "--outdated", "--frozen"],
        cwd=ROOT,
        required_tools=("uv",),
    ),
    Check(
        ecosystem="docs-frontend",
        check_id="pnpm-outdated-docs",
        command=["pnpm", "outdated", "--format", "json"],
        cwd=ROOT / "docs/astro-site",
        required_tools=("pnpm",),
    ),
    Check(
        ecosystem="typescript-binding",
        check_id="npm-outdated-typescript-binding",
        command=["npm", "outdated", "--json"],
        cwd=ROOT / "bindings/typescript",
        required_tools=("npm",),
    ),
    Check(
        ecosystem="rust",
        check_id="cargo-update-dry-run",
        command=["cargo", "update", "--dry-run"],
        cwd=ROOT / "bindings/rust",
        required_tools=("cargo",),
    ),
    Check(
        ecosystem="rust",
        check_id="cargo-outdated",
        command=["cargo", "outdated", "--root-deps-only", "--exit-code", "1"],
        cwd=ROOT / "bindings/rust",
        required_tools=("cargo", "cargo-outdated"),
    ),
    Check(
        ecosystem="r-binding",
        check_id="r-cran-outdated",
        command=[
            "Rscript",
            "-e",
            (
                "description <- read.dcf('bindings/r/DESCRIPTION')[1, ]; "
                "fields <- intersect(c('Depends', 'Imports', 'Suggests'), names(description)); "
                "deps <- unique(unlist(strsplit(paste(description[fields], collapse=','), ','))); "
                "deps <- trimws(gsub(' *[(].*[)]', '', deps)); "
                "deps <- deps[nzchar(deps) & deps != 'R']; "
                "available <- available.packages(repos=Sys.getenv('RSPM', 'https://cloud.r-project.org')); "
                "installed <- installed.packages(); "
                "for (pkg in deps) { "
                "  current <- if (pkg %in% rownames(installed)) installed[pkg, 'Version'] else '<not-installed>'; "
                "  latest <- if (pkg %in% rownames(available)) available[pkg, 'Version'] else '<not-found>'; "
                "  cat(pkg, current, latest, sep='\\t'); cat('\\n'); "
                "}"
            ),
        ],
        cwd=ROOT,
        required_tools=("Rscript",),
    ),
    Check(
        ecosystem="julia-binding",
        check_id="julia-pkg-status-outdated",
        command=[
            "julia",
            "--project=bindings/julia",
            "-e",
            "using Pkg; Pkg.status(; outdated=true)",
        ],
        cwd=ROOT,
        required_tools=("julia",),
    ),
    Check(
        ecosystem="dotnet-binding",
        check_id="dotnet-outdated-runtime-package",
        command=["dotnet", "list", "bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj", "package", "--outdated"],
        cwd=ROOT,
        required_tools=("dotnet",),
    ),
    Check(
        ecosystem="dotnet-binding",
        check_id="dotnet-outdated-test-package",
        command=[
            "dotnet",
            "list",
            "bindings/csharp/Innovate.Kernel.Tests/Innovate.Kernel.Tests.csproj",
            "package",
            "--outdated",
        ],
        cwd=ROOT,
        required_tools=("dotnet",),
    ),
)


def _tool_exists(tool: str) -> bool:
    if tool == "cargo-outdated":
        return "outdated" in subprocess.run(
            ["cargo", "--list"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    return shutil.which(tool) is not None


def _run_check(check: Check, timeout_seconds: int) -> dict[str, Any]:
    missing = [tool for tool in check.required_tools if not _tool_exists(tool)]
    if missing:
        return {
            "check_id": check.check_id,
            "ecosystem": check.ecosystem,
            "status": "tool_missing",
            "command": check.command,
            "cwd": str(check.cwd.relative_to(ROOT)),
            "missing_tools": missing,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }

    result = subprocess.run(
        check.command,
        cwd=check.cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    status = "passed" if result.returncode == 0 else "outdated_or_failed" if check.allow_nonzero else "failed"
    return {
        "check_id": check.check_id,
        "ecosystem": check.ecosystem,
        "status": status,
        "command": check.command,
        "cwd": str(check.cwd.relative_to(ROOT)),
        "missing_tools": [],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_dashboard(timeout_seconds: int = 180) -> dict[str, Any]:
    """Run every dependency freshness check and return a dashboard payload."""
    checks = [_run_check(check, timeout_seconds=timeout_seconds) for check in CHECKS]
    counts: dict[str, int] = {}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/dependency_dashboard.py",
        "policy": "non_mutating_dependency_freshness_dashboard",
        "checks": checks,
        "summary": counts,
    }


def render_markdown(dashboard: dict[str, Any]) -> str:
    """Render a compact Markdown report for uploaded workflow artifacts."""
    lines = [
        "# Dependency Dashboard",
        "",
        f"Generated: `{dashboard['generated_at']}`",
        "",
        "| Ecosystem | Check | Status | Command |",
        "| --- | --- | --- | --- |",
    ]
    for check in dashboard["checks"]:
        command = " ".join(check["command"])
        lines.append(f"| {check['ecosystem']} | {check['check_id']} | {check['status']} | `{command}` |")
    lines.extend(["", "## Details", ""])
    for check in dashboard["checks"]:
        lines.extend(
            [
                f"### {check['ecosystem']}: {check['check_id']}",
                "",
                f"- Status: `{check['status']}`",
                f"- Return code: `{check['returncode']}`",
                f"- Working directory: `{check['cwd']}`",
                "",
                "```text",
                (check["stdout"] or check["stderr"] or "<no output>").strip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_dashboard(dashboard: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dependency-dashboard.json").write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
    (output_dir / "dependency-dashboard.md").write_text(render_markdown(dashboard) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    args = parser.parse_args()

    dashboard = build_dashboard(timeout_seconds=args.timeout_seconds)
    write_dashboard(dashboard, args.output_dir)
    if args.json:
        print(json.dumps(dashboard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
