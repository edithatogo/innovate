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


def verify_font_sources(checkout: Path, paths: tuple[str, ...]) -> None:
    """Check copied files against Git blobs, including ignored/untracked extras."""
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", *paths], cwd=checkout, text=True
    ).splitlines()
    actual = set()
    for relative in paths:
        path = checkout / relative
        if path.is_dir() and not path.is_symlink():
            actual.update(
                str(p.relative_to(checkout)).replace(os.sep, "/")
                for p in path.rglob("*")
                if p.is_file() or p.is_symlink()
            )
        else:
            actual.add(relative)
    if not tracked or set(tracked) != actual:
        raise ValueError(f"Copied font file inventory differs from Git: {checkout}")
    for relative in tracked:
        path = checkout / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(checkout.resolve()):
            raise ValueError(f"Unsafe copied font source: {path}")
        expected = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=checkout)
        if path.read_bytes() != expected:
            raise ValueError(f"Copied font source differs from Git: {path}")


def owned_tex_tool(texlive: Path, name: str) -> Path:
    """Require an executable inside this TeX tree; never search system PATH."""
    root = texlive.resolve(strict=True)
    candidate = root / "bin/x86_64-linux" / name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"Missing owned TeX tool: {candidate}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"TeX tool is outside the owned tree or not executable: {candidate}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "texlive", "inara", "hack", "pandoc"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit(f"Refusing to reuse configuration directory: {args.output}")
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
    try:
        verify_font_sources(hack, ("build/ttf", "LICENSE.md"))
        verify_font_sources(inara, ("fonts/libre-franklin",))
        tex_tools = {
            name: owned_tex_tool(texlive, name) for name in ("luahbtex", "luaotfload-tool", "texlua", "kpsewhich")
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error)) from error
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
            str(tex_tools["luahbtex"]),
            "--ini",
            "--interaction=nonstopmode",
            "--halt-on-error",
            "--jobname=lualatex",
            "--progname=lualatex",
            "lualatex.ini",
        ],
        format_dir,
    )
    run("font-cache", [str(tex_tools["luaotfload-tool"]), "--update", "--force"], inara)
    record["format_sha256"] = digest(format_dir / "lualatex.fmt")
    record["configuration_script_sha256"] = digest(Path(__file__))
    record["status"] = "configured"
    (output / "configuration-receipt.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
