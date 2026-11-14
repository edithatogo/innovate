"""Verify this Astro artifact without trusting historical acceptance JSON.

This static gate does not establish scientific, browser or live acceptance.
External URLs are counted, never requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    """Collect rendered navigation, metadata and basic semantic requirements."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.canonicals: list[str] = []
        self.lang = ""
        self.generators: list[str] = []
        self.description = ""
        self.heading_count = 0
        self.missing_alt = 0
        self.title = ""
        self.in_title = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        if tag == "html":
            self.lang = attrs.get("lang", "")
        if tag == "title":
            self.in_title = True
        if tag == "h1":
            self.heading_count += 1
        if tag == "img" and "alt" not in attrs:
            self.missing_alt += 1
        if tag == "meta":
            if attrs.get("name") == "generator":
                self.generators.append(attrs.get("content", ""))
            if attrs.get("name") == "description":
                self.description = attrs.get("content", "")
        if tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.canonicals.append(attrs.get("href", ""))
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in {"script", "img", "source"} and attrs.get("src"):
            self.links.append(attrs["src"])
        if tag == "link" and "stylesheet" in attrs.get("rel", "").split() and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def digest_files(paths: list[Path], root: Path) -> tuple[str, dict[str, str]]:
    """Hash content and relative filenames in deterministic order."""
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, files


def validate_contract(contract: object) -> None:
    """Reject malformed policies before filesystem access or artifact inspection."""
    if (
        not isinstance(contract, dict)
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != 1
    ):
        raise ValueError("unsupported site-quality contract schema")
    origin = contract.get("origin")
    if not isinstance(origin, str):
        raise ValueError("origin must be an HTTPS origin string")
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.geturl() != origin
        or any(char.isspace() for char in origin)
    ):
        raise ValueError("origin must be HTTPS without a path, credentials, query or fragment")
    # Force validation of malformed/out-of-range ports as well.
    _ = parsed.port

    def normalized_route(value: object) -> bool:
        return (
            isinstance(value, str)
            and value.startswith("/")
            and not any(char in value for char in ("?", "#", "%", "\\", "\x00"))
            and "//" not in value
            and not any(char.isspace() for char in value)
            and not any(part in {".", ".."} for part in value.split("/"))
        )

    base = contract.get("base")
    if not normalized_route(base) or not base.endswith("/"):
        raise ValueError("base must be a normalized path starting and ending with slash")
    for name in ("required_routes", "source_roots", "exclude_source_parts"):
        values = contract.get(name)
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{name} must be a list of nonempty strings")
        if len(values) != len(set(values)) or (name != "exclude_source_parts" and not values):
            raise ValueError(f"{name} must contain unique entries and cannot be empty")
    if not all(normalized_route(route) for route in contract["required_routes"]):
        raise ValueError("required_routes must contain normalized deployment-relative paths")
    for path in contract["source_roots"]:
        if (
            Path(path).is_absolute()
            or "\\" in path
            or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError(f"source_roots must contain normalized relative paths: {path}")
    if any(
        part in {".", ".."} or "/" in part or "\\" in part or "\x00" in part
        for part in contract["exclude_source_parts"]
    ):
        raise ValueError("exclude_source_parts must contain single path components")
    if type(contract.get("max_html_bytes")) is not int or contract["max_html_bytes"] <= 0:
        raise ValueError("max_html_bytes must be a positive integer")


def ensure_contained_path(path: Path, root: Path) -> None:
    """Reject traversal and symlinks in the requested path or any ancestor."""
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path is outside repository root: {path}") from error
    if ".." in relative.parts or not path.resolve().is_relative_to(root):
        raise ValueError(f"path is outside repository root: {relative}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink is not supported: {current.relative_to(root)}")


def source_digest(root: Path, contract: dict) -> tuple[str, dict[str, str]]:
    """Fingerprint configured source inputs, excluding generated output."""
    paths = set()
    excluded = set(contract["exclude_source_parts"])
    for item in contract["source_roots"]:
        path = root / item
        ensure_contained_path(path, root)
        if not path.exists():
            raise ValueError(f"missing source root: {item}")
        for candidate in [path] if path.is_file() else path.rglob("*"):
            if excluded.intersection(candidate.relative_to(root).parts):
                continue
            if candidate.is_symlink():
                raise ValueError(f"source symlink is not supported: {candidate.relative_to(root)}")
            if candidate.is_file():
                ensure_contained_path(candidate, root)
                paths.add(candidate)
    if not paths:
        raise ValueError("source inventory is empty")
    return digest_files(list(paths), root)


def route_for(path: Path, dist: Path, base: str) -> str:
    """Return the deployment path of an emitted HTML file."""
    relative = path.relative_to(dist).as_posix()
    if relative == "index.html":
        return base
    if relative.endswith("/index.html"):
        relative = relative[: -len("index.html")]
    return base + relative


def verify_artifact(root: Path, dist: Path, contract: dict) -> dict:
    """Check built pages, destinations, canonical URLs, search and sitemap."""
    validate_contract(contract)
    root = root.resolve()
    dist = dist if dist.is_absolute() else root / dist
    ensure_contained_path(dist, root)
    errors = []
    origin, base = contract["origin"], contract["base"]
    files = sorted(p for p in dist.rglob("*") if p.is_file() or p.is_symlink())
    safe = []
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(dist.resolve()):
            errors.append({"code": "artifact_symlink", "path": path.relative_to(dist).as_posix()})
        else:
            safe.append(path)
    html = [p for p in safe if p.suffix == ".html"]
    pages = {route_for(p, dist, base): Page(p.read_text(encoding="utf-8")) for p in html}
    if not pages:
        errors.append({"code": "empty_artifact"})
    for path in html:
        route = route_for(path, dist, base)
        page = pages[route]
        canonical_route = base + "404/" if path.relative_to(dist).as_posix() == "404.html" else route
        expected = origin + canonical_route
        if page.canonicals != [expected]:
            errors.append({"code": "canonical", "route": route, "expected": expected, "actual": page.canonicals})
        if not any(generator == "Astro" or generator.startswith("Astro ") for generator in page.generators):
            errors.append({"code": "generator", "route": route})
        if not page.title.strip() or not page.description.strip() or not page.lang:
            errors.append({"code": "page_metadata", "route": route})
        if page.heading_count != 1:
            errors.append({"code": "heading", "route": route, "count": page.heading_count})
        if page.missing_alt:
            errors.append({"code": "image_alt", "route": route, "count": page.missing_alt})
        if path.stat().st_size > contract["max_html_bytes"]:
            errors.append({"code": "html_budget", "route": route, "bytes": path.stat().st_size})

    external = set()
    checked_links = 0

    def destination(url, source):
        parsed = urlsplit(urljoin(origin + source, url))
        if parsed.scheme not in {"http", "https"}:
            return None, None
        if f"{parsed.scheme}://{parsed.netloc}" != origin:
            external.add(url)
            return None, None
        route = unquote(parsed.path)
        if not route.startswith(base):
            errors.append({"code": "outside_base", "source": source, "url": url})
            return None, None
        relative = route[len(base) :]
        candidate = dist / relative
        if not candidate.resolve().is_relative_to(dist.resolve()) or candidate.is_symlink():
            errors.append({"code": "unsafe_destination", "source": source, "url": url})
            return None, None
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file() or candidate.is_symlink():
            errors.append({"code": "broken_link", "source": source, "url": url})
            return None, None
        return candidate, unquote(parsed.fragment)

    for route, page in pages.items():
        for link in page.links:
            target, fragment = destination(link, route)
            if target is None:
                continue
            checked_links += 1
            if fragment and target.suffix == ".html":
                target_page = pages.get(route_for(target, dist, base))
                if target_page is None or fragment not in target_page.ids:
                    errors.append({"code": "broken_fragment", "source": route, "url": link})

    for route in contract["required_routes"]:
        key = base + route.lstrip("/")
        if key not in pages:
            errors.append({"code": "required_route", "route": key})

    search_files = [p for p in safe if "pagefind" in p.relative_to(dist).parts and p.stat().st_size]
    if not any(p.name == "pagefind.js" for p in search_files) or not any(
        p.suffix == ".pf_fragment" for p in search_files
    ):
        errors.append({"code": "search_index", "detail": "Pagefind script and nonempty indexed fragments required"})

    sitemap_routes = set()
    for sitemap in [p for p in safe if p.name.startswith("sitemap") and p.suffix == ".xml"]:
        try:
            xml = ET.fromstring(sitemap.read_text(encoding="utf-8"))
        except ET.ParseError:
            errors.append({"code": "sitemap_parse", "path": sitemap.name})
            continue
        for node in xml.iter():
            if node.tag.rsplit("}", 1)[-1] != "loc" or not node.text:
                continue
            if node.text.endswith(".xml"):
                target, _ = destination(node.text, base)
                if target is None:
                    errors.append({"code": "sitemap_index", "url": node.text})
                continue
            parsed = urlsplit(node.text)
            if f"{parsed.scheme}://{parsed.netloc}" != origin or parsed.path not in pages:
                errors.append({"code": "sitemap_route", "url": node.text})
            else:
                sitemap_routes.add(parsed.path)
    required_sitemap = set(pages) - {base + "404.html", base + "404/"}
    for route in sorted(required_sitemap - sitemap_routes):
        errors.append({"code": "sitemap_missing", "route": route})

    source_hash, source_files = source_digest(root, contract)
    artifact_hash, artifact_files = digest_files(safe, dist)
    return {
        "schema_version": 2,
        "status": "failed" if errors else "passed",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_hash,
        "artifact_sha256": artifact_hash,
        "source_files": source_files,
        "artifact_files": artifact_files,
        "counts": {
            "html_pages": len(pages),
            "checked_local_links": checked_links,
            "external_urls_not_checked": len(external),
            "errors": len(errors),
        },
        "errors": errors,
        "content_complete": False,
        "live_deployment_verified": False,
        "browser_accessibility_verified": False,
        "source_artifact_binding_verified": False,
        "fingerprint_scope": (
            "Contemporaneous source and artifact fingerprints; these hashes alone do not prove "
            "the artifact was built from this source."
        ),
        "scope": "Current static artifact only; API/content/execution/scientific/browser/live gates remain separate.",
    }


def main() -> int:
    """Write a current-build receipt and fail on any observed artifact defect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    dist = args.dist or root / "docs/astro-site/dist"
    contract_path = args.contract or root / "docs/contracts/site-quality.json"
    output = args.output or root / "docs/astro-site/.reports/artifact.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        report = verify_artifact(root, dist, contract)
        report["contract_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, timeout=10)
        report["commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        report = {
            "schema_version": 2,
            "status": "failed",
            "errors": [{"code": "invalid_input", "detail": str(error)}],
            "content_complete": False,
            "live_deployment_verified": False,
            "browser_accessibility_verified": False,
            "source_artifact_binding_verified": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Documentation artifact: {report['status']}; {len(report['errors'])} errors; report: {output}")
        for error in report["errors"][:20]:
            print(json.dumps(error, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
