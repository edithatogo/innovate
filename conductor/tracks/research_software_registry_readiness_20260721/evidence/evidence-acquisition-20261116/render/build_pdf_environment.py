#!/usr/bin/env python3
"""Prepare an owned TeX Live 2024 PDF closure from its historic official archive.

This setup recipe does not modify system TeX, install into another environment,
or qualify an Innovate manuscript. The unmodified Inara wrapper performs renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCH = "x86_64-linux"
REPOSITORY = "https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final"
DATABASE_SHA256 = "60b9f8f8893d591d220d2f4f8dda3f686caa165efe78aa0ccc8dec80d48799aa"
# PDF requirements from pinned Inara's Dockerfile/template plus the LuaLaTeX
# foundation, implicit Lua dependencies, the template's sRGB profile and
# Pandoc's optional spacing/typography packages identified by TeX preflight.
# hyperxmp/fontsetup do not declare all style-file dependencies in tlpdb; the
# final support set below was identified by recursively inspecting their source.
ROOT_PACKAGES = (
    "collection-latex collection-latexrecommended "
    "latex-bin luatex luaotfload luacode luatexbase luatex85 lualatex-math "
    "selnolig fontspec unicode-math amsmath amsfonts pgf geometry fancyhdr "
    "caption colorprofiles etoolbox iftex float xcolor hyperref graphics url algorithmicx "
    "algorithms draftwatermark environ fontsetup hyperxmp latexmk lineno "
    "marginnote newcomputermodern orcidlink preprint seqsplit tcolorbox "
    "titlesec trimspaces xkeyval xstring microtype parskip upquote "
    "ifmtarg oberdiek atenddvi etex-pkg colortbl ltabptch"
).split()


def digest(path: Path, algorithm: str = "sha512") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--reuse-archives", type=Path)
    args = parser.parse_args()
    if args.root.exists() or args.root.is_symlink():
        raise SystemExit(f"Refusing to reuse existing setup root: {args.root}")
    root = args.root.resolve()
    database_bytes = args.database.read_bytes()
    if hashlib.sha256(database_bytes).hexdigest() != DATABASE_SHA256:
        raise SystemExit("TeX Live database does not match the pinned SHA-256; no setup files were written")
    database_path = root / "texlive-2024-final.tlpdb.xz"
    records = {}
    for block in lzma.decompress(database_bytes).decode().split("\n\n"):
        if not block.startswith("name "):
            continue
        fields = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            if key:
                fields.setdefault(key, []).append(value)
        records[fields["name"][0]] = fields
    selected = {}

    def select(name: str) -> None:
        name = name.replace(".ARCH", f".{ARCH}")
        if name in selected:
            return
        fields = records[name]
        if "containerchecksum" not in fields:
            raise ValueError(f"Missing immutable container checksum: {name}")
        selected[name] = {
            "name": name,
            "revision": int(fields["revision"][0]),
            "version": fields.get("catalogue-version", [None])[0],
            "license": fields.get("catalogue-license", [None])[0],
            "relocated": fields.get("relocated", ["0"])[0] == "1",
            "archive_size": int(fields["containersize"][0]),
            "archive_sha512": fields["containerchecksum"][0],
            "url": f"{REPOSITORY}/archive/{name}.tar.xz",
            "dependencies": [d.replace(".ARCH", f".{ARCH}") for d in fields.get("depend", [])],
        }
        for dependency in selected[name]["dependencies"]:
            select(dependency)

    for name in ROOT_PACKAGES:
        select(name)
    packages = [selected[name] for name in sorted(selected)]
    # Validate the complete pinned database and package selection before writing.
    root.mkdir(parents=True, exist_ok=False)
    archives = root / "archives"
    archives.mkdir()
    destination = root / "texlive"
    database_path.write_bytes(database_bytes)
    manifest = {
        "schema_version": 1,
        "scope": "local Linux x86_64 LuaLaTeX PDF build; not complete Inara container or other formats",
        "repository": REPOSITORY,
        "database_url": f"{REPOSITORY}/tlpkg/texlive.tlpdb.xz",
        "database_sha256": hashlib.sha256(database_bytes).hexdigest(),
        "architecture": ARCH,
        "root_packages": ROOT_PACKAGES,
        "package_count": len(packages),
        "total_archive_bytes": sum(p["archive_size"] for p in packages),
        "packages": packages,
    }
    (root / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Resolved {len(packages)} packages, {manifest['total_archive_bytes']} compressed bytes", flush=True)

    def acquire(package: dict) -> str:
        archive = archives / (package["name"] + ".tar.xz")
        if not archive.exists() and args.reuse_archives:
            cached = args.reuse_archives / archive.name
            if cached.is_file() and digest(cached) == package["archive_sha512"]:
                shutil.copy2(cached, archive)
        if not archive.exists():
            partial = archive.with_suffix(".part")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "60",
                    package["url"],
                    "--output",
                    str(partial),
                ],
                check=True,
            )
            partial.replace(archive)
        if archive.stat().st_size != package["archive_size"] or digest(archive) != package["archive_sha512"]:
            raise ValueError(f"Official database checksum/size mismatch: {archive.name}")
        return package["name"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(acquire, package) for package in packages]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(packages):
                print(f"Verified archives {count}/{len(packages)}", flush=True)
    destination.mkdir()
    for package in packages:
        archive = archives / (package["name"] + ".tar.xz")
        with tarfile.open(archive, "r:xz") as bundle:
            for member in bundle:
                # TL's relocated archives omit texmf-dist/ from runtime paths.
                # Package metadata still belongs to the installation's tlpkg/.
                target = destination
                if package["relocated"] and not member.name.startswith("tlpkg/"):
                    target = destination / "texmf-dist"
                bundle.extract(member, target, filter="data")
    print(f"Extracted verified PDF toolchain: {destination}", flush=True)


if __name__ == "__main__":
    main()
