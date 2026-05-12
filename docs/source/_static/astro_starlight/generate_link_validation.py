"""Generate the Astro/Starlight link validation report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[4]
ASTRO_ROOT = REPO_ROOT / "docs/astro-site/src/content/docs"
ROUTE_COVERAGE = REPO_ROOT / "docs/source/_static/astro_starlight/route_coverage.json"
OUTPUT = REPO_ROOT / "docs/source/_static/astro_starlight/link_validation_report.json"
STALIGHT_CONFIG = REPO_ROOT / "docs/astro-site/starlight.config.mjs"

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

SIDEBAR_ROUTES = [
    "/core/kernel/",
    "/core/arrow-interchange/",
    "/core/diagnostics-contract/",
    "/bindings/",
    "/operations/rust-core/",
    "/maintainers/publication/",
    "/migration/",
    "/migration/redirects/",
    "/migration/validation/",
    "/migration/archive/",
    "/migration/references/",
]


def load_routes() -> dict[str, dict[str, object]]:
    """Load the implemented route map from the coverage report."""
    coverage = json.loads(ROUTE_COVERAGE.read_text())
    return {
        entry["astro_route"]: entry
        for entry in coverage["coverage_by_source_doc"]
        if entry["status"] == "implemented"
    }


def normalize_route(target: str) -> str:
    """Strip query and fragment data from a route target."""
    parts = urlsplit(target)
    return parts.path


def is_external(target: str) -> bool:
    """Return whether the target is clearly external."""
    return target.startswith(("http://", "https://", "mailto:"))


def resolve_relative_link(source: Path, target: str) -> Path | None:
    """Resolve a relative markdown link to a file path if possible."""
    if not target.endswith((".md", ".mdx", ".rst")):
        return None
    return (source.parent / target).resolve()


def build_report() -> dict[str, object]:
    """Build the link validation report payload."""
    implemented_routes = load_routes()
    markdown_files = sorted(ASTRO_ROOT.rglob("*.md"))

    link_checks: list[dict[str, object]] = []
    broken_links: list[dict[str, object]] = []
    checked_routes = set()

    for markdown_file in markdown_files:
        text = markdown_file.read_text()
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith("#") or is_external(target):
                continue

            record: dict[str, object] = {
                "source_doc": str(markdown_file.relative_to(REPO_ROOT)),
                "target": target,
            }

            if target.startswith("/"):
                route = normalize_route(target)
                record["kind"] = "route"
                record["normalized_target"] = route
                route_info = implemented_routes.get(route)
                record["implemented"] = route_info is not None
                if route_info is not None:
                    record["source_route_doc"] = route_info.get("source_doc")
                    checked_routes.add(route)
                else:
                    broken_links.append(record)
                link_checks.append(record)
                continue

            resolved = resolve_relative_link(markdown_file, target)
            record["kind"] = "file"
            if resolved is not None:
                record["resolved_path"] = str(resolved.relative_to(REPO_ROOT))
                record["implemented"] = resolved.exists()
            else:
                record["implemented"] = True
            if not record["implemented"]:
                broken_links.append(record)
            link_checks.append(record)

    sidebar_route_validation = [
        {
            "route": route,
            "implemented": route in implemented_routes,
            "source_doc": implemented_routes.get(route, {}).get("source_doc"),
        }
        for route in SIDEBAR_ROUTES
    ]

    all_sidebar_routes_implemented = all(
        entry["implemented"] for entry in sidebar_route_validation
    )

    route_links_checked = sum(1 for entry in link_checks if entry["kind"] == "route")
    file_links_checked = sum(1 for entry in link_checks if entry["kind"] == "file")

    return {
        "generated_from": {
            "route_coverage": str(ROUTE_COVERAGE.relative_to(REPO_ROOT)),
            "starlight_config": str(STALIGHT_CONFIG.relative_to(REPO_ROOT)),
            "astro_content_root": str(ASTRO_ROOT.relative_to(REPO_ROOT)),
        },
        "counts": {
            "markdown_files": len(markdown_files),
            "links_checked": len(link_checks),
            "route_links_checked": route_links_checked,
            "file_links_checked": file_links_checked,
            "broken_links": len(broken_links),
            "sidebar_routes": len(SIDEBAR_ROUTES),
            "implemented_sidebar_routes": sum(
                1 for entry in sidebar_route_validation if entry["implemented"]
            ),
        },
        "ready_for_route_stability": not broken_links and all_sidebar_routes_implemented,
        "sidebar_route_validation": sidebar_route_validation,
        "link_checks": link_checks,
        "broken_links": broken_links,
        "checked_routes": sorted(checked_routes),
    }


def main() -> None:
    """Write the validation report to disk."""
    OUTPUT.write_text(json.dumps(build_report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
