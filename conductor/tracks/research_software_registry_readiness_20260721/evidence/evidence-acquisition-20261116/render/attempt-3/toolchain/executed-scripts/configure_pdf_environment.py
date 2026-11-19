#!/usr/bin/env python3
"""Configure owned fonts, LuaLaTeX format and cache for the pinned draft render."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "texlive", "inara", "hack", "pandoc"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    output, texlive, inara, hack, pandoc = (
        getattr(args, key).resolve() for key in ("output", "texlive", "inara", "hack", "pandoc")
    )
    if output.exists():
        raise SystemExit(f"Refusing to reuse configuration directory: {output}")
    for path, revision in (
        (inara, "88ee5d656f147d801448ce77ad8375619d383df6"),
        (hack, "b47d21f808fbf4784565feccde2213ef2af924b3"),
    ):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if actual != revision:
            raise SystemExit(f"Unexpected source revision for {path}: {actual}")
    if subprocess.check_output([str(pandoc), "--version"], text=True).splitlines()[0] != "pandoc 3.6.4":
        raise SystemExit("Pandoc 3.6.4 is required")
    fonts = output / "texmf-local/fonts/truetype"
    shutil.copytree(hack / "build/ttf", fonts / "hack")
    shutil.copy2(hack / "LICENSE.md", fonts / "hack/LICENSE.md")
    shutil.copytree(inara / "fonts/libre-franklin", fonts / "libre-franklin")
    for part in ("var", "config", "fonts"):
        (output / part).mkdir(parents=True)
    env = {
        "PATH": f"{texlive}/bin/x86_64-linux:{pandoc.parent}:" + os.environ["PATH"],
        "TEXMFCNF": str(texlive / "texmf-dist/web2c"),
        "TEXMF": "{"
        + ",".join(str(p) for p in (output / "var", output / "config", output / "texmf-local", texlive / "texmf-dist"))
        + "}",
        "TEXMFVAR": str(output / "var"),
        "TEXMFCONFIG": str(output / "config"),
        "TEXMFHOME": str(output / "texmf-local"),
        "TEXMFLOCAL": str(output / "texmf-local"),
        "TEXMFCACHE": ".",
        "VARTEXFONTS": str(output / "fonts"),
        "OSFONTDIR": f"{fonts}/hack;{fonts}/libre-franklin",
        "SOURCE_DATE_EPOCH": "1794787200",
    }
    (output / "render-env.json").write_text(json.dumps(env, indent=2) + "\n")
    active = os.environ.copy()
    active.update(env)
    record = {
        "scope": "Owned Linux x86_64 local PDF environment; no author/scientific/submission acceptance",
        "platform": platform.platform(),
        "pandoc_sha256": digest(pandoc),
        "font_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(fonts.rglob("*.ttf"))},
        "commands": [],
    }

    def run(stage: str, command: list[str], cwd: Path) -> None:
        log = output / (stage + ".log")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=cwd, env=active, stdout=stream, stderr=subprocess.STDOUT)
        record["commands"].append(
            {
                "stage": stage,
                "argv": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "log_sha256": digest(log),
            }
        )
        (output / "configuration-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{stage}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    format_dir = output / "var/web2c/luahbtex"
    format_dir.mkdir(parents=True)
    run(
        "format-build",
        [
            "luahbtex",
            "--ini",
            "--interaction=nonstopmode",
            "--halt-on-error",
            "--jobname=lualatex",
            "--progname=lualatex",
            "lualatex.ini",
        ],
        format_dir,
    )
    run("font-cache", ["luaotfload-tool", "--update", "--force"], inara)
    record["format_sha256"] = digest(format_dir / "lualatex.fmt")
    record["configuration_script_sha256"] = digest(Path(__file__))
    record["status"] = "configured"
    (output / "configuration-receipt.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
