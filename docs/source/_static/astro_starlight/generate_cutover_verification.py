"""Generate the Astro/Starlight cutover verification report."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTENT_INVENTORY = REPO_ROOT / "docs/source/_static/astro_starlight/content_inventory.json"
REDIRECT_INVENTORY = REPO_ROOT / "docs/source/_static/astro_starlight/redirect_inventory.json"
OUTPUT = REPO_ROOT / "docs/source/_static/astro_starlight/cutover_verification.json"


def build_report() -> dict[str, object]:
    """Build a report that proves the content and redirect inventories align."""
    content_inventory = json.loads(CONTENT_INVENTORY.read_text())
    redirect_inventory = json.loads(REDIRECT_INVENTORY.read_text())

    redirect_by_source = {entry["source_doc"]: entry for entry in redirect_inventory}
    matched_entries: list[dict[str, object]] = []
    content_only: list[dict[str, str]] = []
    redirect_only: list[dict[str, str]] = []

    for entry in content_inventory:
        redirect_entry = redirect_by_source.pop(entry["source_doc"], None)
        if redirect_entry is None:
            content_only.append(
                {
                    "source_doc": entry["source_doc"],
                    "astro_route": entry["astro_route"],
                }
            )
            continue

        matched_entries.append(
            {
                "source_doc": entry["source_doc"],
                "sphinx_path": entry["sphinx_path"],
                "astro_route": entry["astro_route"],
                "redirect_type": redirect_entry["redirect_type"],
                "routes_match": entry["astro_route"] == redirect_entry["astro_route"],
                "redirect_type_ok": redirect_entry["redirect_type"] == "temporary-forward",
            }
        )

    for remaining in redirect_by_source.values():
        redirect_only.append(
            {
                "source_doc": remaining["source_doc"],
                "astro_route": remaining["astro_route"],
            }
        )

    all_routes_match = all(item["routes_match"] for item in matched_entries)
    all_redirects_forward = all(item["redirect_type_ok"] for item in matched_entries)
    ready_for_cutover = (
        not content_only
        and not redirect_only
        and all_routes_match
        and all_redirects_forward
        and len(matched_entries) == len(content_inventory)
    )

    return {
        "generated_from": {
            "content_inventory": str(CONTENT_INVENTORY.relative_to(REPO_ROOT)),
            "redirect_inventory": str(REDIRECT_INVENTORY.relative_to(REPO_ROOT)),
        },
        "counts": {
            "content_entries": len(content_inventory),
            "redirect_entries": len(redirect_inventory),
            "matched_entries": len(matched_entries),
            "content_only": len(content_only),
            "redirect_only": len(redirect_only),
        },
        "ready_for_cutover": ready_for_cutover,
        "matched_entries": matched_entries,
        "content_only": content_only,
        "redirect_only": redirect_only,
    }


def main() -> None:
    """Write the verification report to disk."""
    OUTPUT.write_text(json.dumps(build_report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
