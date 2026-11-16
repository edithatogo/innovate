"""Check removal of obsolete sources and the migration notices for their URLs.

The source inventory preserves hashes and immutable Git links without keeping
unsupported examples in the maintained documentation. It is not a scientific
acceptance or feature-parity record. No network requests are made by this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("docs/contracts/retired-documentation.json")
SITE = Path("docs/astro-site/src/content/docs")
CATEGORIES = {
    "historical_page",
    "versioned_page",
    "sphinx_source",
    "legacy_notebook",
    "legacy_guide",
    "legacy_generator",
}
RELATIONSHIPS = {
    "current_replacement",
    "partial_replacement",
    "unsupported_interface",
    "no_qualified_replacement",
    "historical_version",
}


def _relative(value: object) -> str:
    """Return a normalized repository path or reject traversal and URL syntax."""
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or any(c in value for c in "\\?#\0")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"unsafe repository path: {value!r}")
    return value


def _route(value: object) -> str:
    """Validate a deployment-relative HTML route, including the site base."""
    if (
        not isinstance(value, str)
        or not value.startswith("/innovate/")
        or not value.endswith("/")
        or any(c.isspace() for c in value)
        or any(c in value for c in "\\?#%\0")
        or "//" in value
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"unsafe documentation route: {value!r}")
    return value


def route_for_source(relative: str) -> str:
    """Map an unversioned Markdown content path to its current Astro route."""
    stem = re.sub(r"\.mdx?$", "", relative)
    stem = re.sub(r"(^|/)index$", "", stem).rstrip("/")
    return "/innovate/" + (stem + "/" if stem else "")


class RetirementPage(HTMLParser):
    """Collect only the notice properties that the general HTML gate cannot infer."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.links: set[str] = set()
        self.sources: list[str] = []
        self.robots: list[str] = []
        self.searchable = False
        self.code_blocks = 0
        self.version_selectors = 0
        self.text: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Read retirement identity, search exclusion and navigable destinations."""
        values = dict(attrs)
        if "data-pagefind-body" in values:
            self.searchable = True
        if "data-retired-source" in values:
            self.sources.append(values["data-retired-source"] or "")
        if tag == "a" and values.get("href"):
            self.links.add(values["href"])
        if tag == "meta" and values.get("name") == "robots":
            self.robots.append(values.get("content", ""))
        if tag in {"pre", "code"}:
            self.code_blocks += 1
        if tag == "starlight-version-select" or (
            tag == "select" and "version" in str(values.get("aria-label", "")).lower()
        ):
            self.version_selectors += 1

    def handle_data(self, data: str) -> None:
        """Retain visible notice text for the bounded marker check."""
        self.text.append(data)


def audit(root: Path, contract: dict, dist: Path | None = None, *, history: bool = False) -> dict:
    """Validate the manifest, absent sources, current routes and optional HTML.

    Args:
        root: Repository root; no manifest path may escape this directory.
        contract: Parsed retirement manifest with immutable source identities.
        dist: Optional current Astro output for notice and search checks.
        history: Verify hashes against Git objects when the source commit exists.

    Returns:
        A structural receipt. A passing result makes no runtime or scientific claim.

    Raises:
        ValueError: The manifest is malformed or a source, route or notice is invalid.
        OSError: A required local input cannot be read.
        subprocess.SubprocessError: Requested Git history verification fails.
    """
    root = root.resolve()
    if (
        not isinstance(contract, dict)
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != 1
        or set(contract) != {"schema_version", "retired_at", "source_commit", "policy", "entries"}
    ):
        raise ValueError("unsupported retirement manifest")
    commit = contract["source_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source_commit must identify an immutable Git commit")
    if (
        not isinstance(contract["retired_at"], str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", contract["retired_at"])
        or not isinstance(contract["policy"], str)
        or not contract["policy"].strip()
    ):
        raise ValueError("retirement date and policy are required")
    date.fromisoformat(contract["retired_at"])
    entries = contract["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("retirement inventory cannot be empty")
    content = json.loads((root / "docs/contracts/content-quality.json").read_text())
    current = {route_for_source(p) for p, item in content["pages"].items() if item["kind"] != "historical"}
    manifest = root / content["generated_api"]["manifest"]
    generated = json.loads(manifest.read_text())
    for relative in generated["pages"]:
        current.add(route_for_source((manifest.parent / relative).relative_to(root / SITE).as_posix()))
    seen_paths: set[str] = set()
    seen_routes: set[str] = set()
    notices = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("retirement entries must be objects")
        fields = {"path", "category", "sha256", "git_blob", "history_url", "replacement", "scope"}
        if "route" in entry:
            fields |= {"route", "title", "relationship"}
        if set(entry) != fields:
            raise ValueError("unexpected or missing retirement entry fields")
        name = _relative(entry["path"])
        if name in seen_paths:
            raise ValueError(f"duplicate retirement source: {name}")
        seen_paths.add(name)
        source = root / name
        if not source.resolve().is_relative_to(root) or source.is_symlink():
            raise ValueError(f"unsafe retired source: {name}")
        if source.exists():
            raise ValueError(f"retired content has returned: {name}")
        if entry["category"] not in CATEGORIES or not isinstance(entry["scope"], str) or not entry["scope"].strip():
            raise ValueError(f"invalid retirement category or scope: {name}")
        web_source = entry["category"] in {"historical_page", "versioned_page"}
        if web_source != ("route" in entry):
            raise ValueError(f"web sources require a migration route; other sources cannot have one: {name}")
        if web_source:
            if not name.startswith(SITE.as_posix() + "/") or not re.search(r"\.mdx?$", name):
                raise ValueError(f"web retirement source must be a Markdown page: {name}")
            relative = name.removeprefix(SITE.as_posix() + "/")
            if (entry["category"] == "versioned_page") != relative.startswith("latest/") or entry[
                "route"
            ] != route_for_source(relative):
                raise ValueError(f"retirement route is not bound to its source: {name}")
        for field, length in (("sha256", 64), ("git_blob", 40)):
            if not isinstance(entry[field], str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", entry[field]):
                raise ValueError(f"invalid {field}: {name}")
        expected_url = f"https://github.com/edithatogo/innovate/blob/{commit}/{quote(name, safe='/')}"
        if entry["history_url"] != expected_url:
            raise ValueError(f"history URL is not bound to the recorded source: {name}")
        target = _route(entry["replacement"])
        if target not in current:
            raise ValueError(f"retirement target is not current: {target}")
        if history:
            raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root, timeout=15)
            blob = subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{name}"], cwd=root, text=True, timeout=15
            ).strip()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"] or blob != entry["git_blob"]:
                raise ValueError(f"retired source identity changed: {name}")
        if "route" not in entry:
            continue
        route = _route(entry["route"])
        if route in seen_routes or route in current:
            raise ValueError(f"duplicate or current retirement route: {route}")
        seen_routes.add(route)
        if (
            entry["relationship"] not in RELATIONSHIPS
            or not entry["title"]
            or entry["category"] not in {"historical_page", "versioned_page"}
        ):
            raise ValueError(f"invalid web migration notice: {route}")
        if dist is not None:
            path = dist / route.removeprefix("/innovate/") / "index.html"
            if path.is_symlink() or not path.resolve().is_relative_to(dist.resolve()):
                raise ValueError(f"unsafe notice artifact: {route}")
            page = RetirementPage(path.read_text())
            if (
                page.sources != [name]
                or page.searchable
                or page.code_blocks
                or page.version_selectors
                or page.robots != ["noindex, follow"]
                or not {target, expected_url}.issubset(page.links)
                or "This documentation has been retired" not in " ".join(page.text)
            ):
                raise ValueError(f"invalid rendered retirement notice: {route}")
            notices += 1
    for directory in (root / SITE / "latest", root / "old_notebooks"):
        if any(p.is_file() for p in directory.rglob("*")):
            raise ValueError(f"retired content directory is nonempty: {directory.relative_to(root)}")
    if any(
        p.is_file() and "_static" not in p.relative_to(root / "docs/source").parts
        for p in (root / "docs/source").rglob("*")
    ):
        raise ValueError("Sphinx source has returned outside retained evidence")
    return {
        "schema_version": 1,
        "status": "passed",
        "removed_sources": len(entries),
        "migration_routes": len(seen_routes),
        "rendered_notices_checked": notices,
        "git_history_checked": history,
        "scientific_acceptance": False,
    }


def main() -> int:
    """Run the source-only gate or verify notices from the current static build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = audit(args.root, json.loads((args.root / CONTRACT).read_text()), args.dist, history=args.history)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        report = {"schema_version": 1, "status": "failed", "error": str(error), "scientific_acceptance": False}
    print(json.dumps(report, indent=2) if args.json else f"Retired documentation: {report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
