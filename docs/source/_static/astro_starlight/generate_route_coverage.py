"""Generate the Astro/Starlight route coverage report.

The report combines the migration inventory with a small set of migration
support docs that have concrete Astro counterparts. The goal is to make the
current coverage explicit during the parallel-run window.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ASTRO_CONTENT_ROOT = REPO_ROOT / "docs/astro-site/src/content/docs"
CONTENT_INVENTORY = REPO_ROOT / "docs/source/_static/astro_starlight/content_inventory.json"
OUTPUT = REPO_ROOT / "docs/source/_static/astro_starlight/route_coverage.json"

SUPPORT_DOCS = [
    {
        "source_doc": "docs/source/astro_starlight_migration.rst",
        "astro_route": "/migration/",
        "astro_content_path": "docs/astro-site/src/content/docs/migration/index.md",
        "status": "implemented",
        "coverage_source": "migration-support",
    },
    {
        "source_doc": "docs/astro-site/src/content/docs/migration/redirects.md",
        "astro_route": "/migration/redirects/",
        "astro_content_path": "docs/astro-site/src/content/docs/migration/redirects.md",
        "status": "implemented",
        "coverage_source": "migration-support",
    },
    {
        "source_doc": "docs/astro-site/src/content/docs/migration/archive.md",
        "astro_route": "/migration/archive/",
        "astro_content_path": "docs/astro-site/src/content/docs/migration/archive.md",
        "status": "implemented",
        "coverage_source": "migration-support",
    },
    {
        "source_doc": "docs/astro-site/src/content/docs/migration/references.md",
        "astro_route": "/migration/references/",
        "astro_content_path": "docs/astro-site/src/content/docs/migration/references.md",
        "status": "implemented",
        "coverage_source": "migration-support",
    },
]


def candidate_paths(route: str) -> list[Path]:
    """Return plausible Astro content paths for a route."""
    if route == "/":
        return [ASTRO_CONTENT_ROOT / "index.md"]

    stripped = route.strip("/")
    parts = stripped.split("/")
    candidates = [
        ASTRO_CONTENT_ROOT.joinpath(*parts).with_suffix(".md"),
        ASTRO_CONTENT_ROOT.joinpath(*parts, "index.md"),
    ]
    return candidates


def resolve_inventory_entry(entry: dict[str, str]) -> dict[str, str]:
    """Annotate an inventory entry with its coverage status."""
    route = entry["astro_route"]
    astro_path = next((path for path in candidate_paths(route) if path.exists()), None)
    status = "implemented" if astro_path is not None else "planned"

    result = {
        "source_doc": entry["source_doc"],
        "astro_route": route,
        "status": status,
        "coverage_source": "inventory",
    }
    if astro_path is not None:
        result["astro_content_path"] = str(astro_path.relative_to(REPO_ROOT))
    return result


def build_report() -> dict[str, object]:
    """Build the route coverage report payload."""
    inventory = json.loads(CONTENT_INVENTORY.read_text())
    coverage = [resolve_inventory_entry(entry) for entry in inventory]

    for support_doc in SUPPORT_DOCS:
        astro_path = REPO_ROOT / support_doc["astro_content_path"]
        status = "implemented" if astro_path.exists() else "planned"
        record = dict(support_doc)
        record["status"] = status
        if status == "implemented":
            record["astro_content_path"] = str(astro_path.relative_to(REPO_ROOT))
        coverage.append(record)

    counts = {
        "total": len(coverage),
        "implemented": sum(1 for entry in coverage if entry["status"] == "implemented"),
        "planned": sum(1 for entry in coverage if entry["status"] == "planned"),
    }

    return {
        "generated_from": {
            "content_inventory": str(CONTENT_INVENTORY.relative_to(REPO_ROOT)),
            "astro_content_root": str(ASTRO_CONTENT_ROOT.relative_to(REPO_ROOT)),
            "support_docs": [doc["source_doc"] for doc in SUPPORT_DOCS],
        },
        "counts": counts,
        "coverage_by_source_doc": coverage,
    }


def main() -> None:
    """Write the coverage report to disk."""
    OUTPUT.write_text(json.dumps(build_report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
