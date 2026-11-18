#!/usr/bin/env python3
"""Parse the bibliography and build an explicitly draft JOSS artifact with Inara.

This wrapper installs nothing, never submits a paper, and leaves a receipt even
when PDF compilation fails. The manuscript integrity check runs before parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

INARA_REVISION = "88ee5d656f147d801448ce77ad8375619d383df6"
PANDOC_VERSION = "3.6.4"
PACKET = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_command(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def safe_make_path(path: Path) -> Path:
    """Upstream Make recipes expand paths without shell quoting."""
    path = path.resolve()
    if not re.fullmatch(r"[/A-Za-z0-9_.-]+", str(path)):
        raise ValueError(f"Inara Make paths must exclude spaces/metacharacters: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inara", required=True, type=Path, help="Pinned Inara checkout")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    parser.add_argument("--pandoc", default="pandoc", help="Pandoc 3.6.4 executable")
    parser.add_argument("--source-date-epoch", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        inara = safe_make_path(args.inara)
        output = safe_make_path(args.output_dir)
        article = safe_make_path(PACKET / "paper.md")
        executable = shutil.which(args.pandoc)
        if executable is None:
            raise ValueError(f"Pandoc executable not found: {args.pandoc}")
        pandoc = safe_make_path(Path(executable))
        if output.exists():
            raise ValueError(f"Refusing to reuse or overwrite an output directory: {output}")
        revision = read_command(["git", "rev-parse", "HEAD"], inara)
        if revision != INARA_REVISION:
            raise ValueError(f"Expected Inara {INARA_REVISION}; found {revision}")
        status = read_command(["git", "status", "--porcelain"], inara)
        # A process-local Lua font cache may live in this isolated checkout.
        changes = [line for line in status.splitlines() if line != "?? luatex-cache/"]
        if changes:
            raise ValueError("Inara checkout contains changes: " + "; ".join(changes))
        pandoc_version = read_command([str(pandoc), "--version"]).splitlines()[0]
        if pandoc_version != f"pandoc {PANDOC_VERSION}":
            raise ValueError(f"Expected Pandoc {PANDOC_VERSION}; found {pandoc_version}")
        epoch = args.source_date_epoch
        if epoch is None:
            epoch = int(os.environ.get("SOURCE_DATE_EPOCH", str(int(time.time()))))
        if epoch < 0:
            raise ValueError("SOURCE_DATE_EPOCH must not be negative")
        input_names = ("paper.md", "paper.bib", "source-snapshot.json")
        inputs = {name: digest(PACKET / name) for name in input_names}
        software_revision = json.loads((PACKET / "source-snapshot.json").read_text())["software_revision"]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        parser.error(str(error))

    output.mkdir(parents=True)
    env = os.environ.copy()
    # Inara's self-citation filter launches `pandoc` by name as a subprocess.
    env["PATH"] = str(pandoc.parent) + os.pathsep + env.get("PATH", "")
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "submission_ready": False,
        "scientific_or_release_qualification": False,
        "visual_review": "not_performed",
        "inara": {"url": "https://github.com/openjournals/inara", "revision": revision},
        "repository_head_observed": read_command(["git", "rev-parse", "HEAD"], PACKET),
        "render_script_sha256": digest(Path(__file__).resolve()),
        "input_sha256": inputs,
        "software_revision": software_revision,
        "source_date_epoch": epoch,
        "draft_mode": True,
        "commands": [],
        "tools": {"pandoc": {"path": str(pandoc), "version": pandoc_version}},
        "tex_environment": {
            key: env[key]
            for key in (
                "TEXMF",
                "TEXMFVAR",
                "TEXMFCONFIG",
                "TEXMFCACHE",
                "VARTEXFONTS",
                "OSFONTDIR",
            )
            if key in env
        },
        "checks": {},
        "output_sha256": {},
    }
    for tool in ("make", "latexmk", "lualatex"):
        path = shutil.which(tool, path=env["PATH"])
        if path:
            result = subprocess.run([path, "--version"], text=True, capture_output=True, env=env)
            receipt["tools"][tool] = {
                "path": path,
                "version_output": (result.stdout + result.stderr).strip(),
                "version_exit_code": result.returncode,
            }
        else:
            receipt["tools"][tool] = {"path": None}

    def run(stage: str, command: list[str]) -> None:
        with (output / "render.log").open("a", encoding="utf-8") as log:
            log.write(f"\n## {stage}\n{json.dumps(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=inara, env=env, text=True, stdout=log, stderr=log)
        receipt["commands"].append({"stage": stage, "cwd": str(inara), "argv": command})
        receipt["checks"][stage] = {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
        }
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}); see render.log")

    exit_code = 1
    try:
        repository = PACKET.parents[3]
        run(
            "source_integrity",
            [sys.executable, str(repository / "scripts/check_joss_readiness.py"), "--root", str(repository), "--json"],
        )
        run(
            "bibtex_parse",
            [
                str(pandoc),
                "--fail-if-warnings",
                "--from=bibtex",
                "--to=csljson",
                str(PACKET / "paper.bib"),
                "--output=" + str(output / "bibliography.json"),
            ],
        )
        bibliography = json.loads((output / "bibliography.json").read_text())
        receipt["checks"]["bibtex_parse"]["entry_ids"] = sorted(item["id"] for item in bibliography)
        # These are the official entrypoint's draft metadata/template flags.
        # Do not rely on the Makefile's article-info default to enable draft mode.
        pandoc_make = str(pandoc) + " --fail-if-warnings --metadata=draft:true --variable=draft:true"
        for format_name in ("tex", "pdf"):
            run(
                "official_" + format_name,
                [
                    "make",
                    format_name,
                    "ARTICLE=" + str(article),
                    "TARGET_FOLDER=" + str(output),
                    "PANDOC=" + pandoc_make,
                ],
            )
            artifact = output / ("paper." + format_name)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise RuntimeError(f"{format_name} command succeeded without a non-empty artifact")
            if format_name == "tex":
                source = artifact.read_text()
                required = (
                    "\\usepackage{draftwatermark}",
                    "\\usepackage{lineno}",
                    "Working draft: not author-approved or submission-ready.",
                )
                if not all(token in source for token in required):
                    raise RuntimeError("Official TeX lacks draft watermark, line numbers or working-draft notice")
                receipt["checks"]["draft_tex_markers"] = {"status": "passed"}
        receipt["status"] = "rendered_draft_visual_review_pending"
        exit_code = 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        receipt["status"] = "blocked"
        receipt["error"] = str(error)
    finally:
        try:
            final_inputs = {name: digest(PACKET / name) for name in input_names}
        except OSError as error:
            final_inputs = None
            receipt["input_recheck_error"] = str(error)
        if inputs != final_inputs:
            receipt["status"] = "blocked"
            receipt["error"] = "Manuscript inputs changed during rendering; do not use these outputs"
            exit_code = 1
        for name in ("bibliography.json", "paper.tex", "paper.pdf", "render.log"):
            path = output / name
            if path.is_file():
                receipt["output_sha256"][name] = digest(path)
        log = output / "render.log"
        if log.is_file():
            receipt["diagnostic_lines"] = [line for line in log.read_text().splitlines() if line.startswith("! ")][:10]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{receipt['status']}: {output / 'receipt.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
