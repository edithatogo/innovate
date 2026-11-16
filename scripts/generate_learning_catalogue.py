"""Build a static learning catalogue from native discovery and content contracts.

This reads source metadata without importing or executing Innovate. Advertised
model fields are preserved verbatim; they are not a qualification result or a
Cartesian product of supported models, variants, quantities and operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = "rust/crates/innovate-core/src/discovery.json"
CONTENT = "docs/contracts/content-quality.json"
PAGES = "docs/astro-site/src/content/docs"
GENERATOR = "scripts/generate_learning_catalogue.py"
OUTPUT = "docs/contracts/learning-catalogue.json"

# Editorial orientation only. Capability fields always come from discovery.json.
MODEL_GUIDANCE = {
    "bass": ("Bass", "Explore adoption driven by external influence and imitation."),
    "logistic": ("Logistic", "Describe an S-shaped level trajectory with a midpoint and saturation scale."),
    "gompertz": ("Gompertz", "Describe an asymmetric trajectory approaching a saturation level."),
    "fisher_pry": ("Fisher–Pry", "Explore a substitution share between technologies."),
    "norton_bass": ("Norton–Bass", "Compare successive technology generations and their transfer conventions."),
    "lotka_volterra": ("Lotka–Volterra", "Explore two interacting shares with separate capacities."),
    "multi_product": ("Multi-product competition", "Explore interacting product levels and market potentials."),
    "complementary_goods": ("Complementary goods", "Explore complementary interactions between product levels."),
    "lock_in": ("Lock-in", "Examine the native historical lock-in and capacity conventions."),
    "network_diffusion": (
        "Network diffusion",
        "Explore deterministic spillover; this model key is distinct from the Bass Kairos network engine.",
    ),
    "policy_hazard": ("Policy hazard", "Inspect a bounded policy-timing level multiplier and its interpretation."),
    "latent_process": ("Latent process", "Inspect a smoothed residual path around a base trajectory."),
    "regime_switching": ("Regime switching", "Compare piecewise trajectories under explicit regime conventions."),
    "hierarchical": ("Hierarchical", "Compare grouped point trajectories and their centering conventions."),
    "mixture": ("Mixture", "Distinguish weighted component trajectories from Gaussian residual-mixture fitting."),
    "composite": ("Composite", "Explore coupled growth trajectories with the native ODE conventions."),
}


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently replacing declarations."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _relative(value: str) -> PurePosixPath:
    """Require an unambiguous repository-relative source path."""
    if not isinstance(value, str) or not value or re.search(r"[\\?#\x00]", value):
        raise ValueError(f"unsafe source path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe source path: {value!r}")
    return path


def _read(root: Path, relative: str, sources: dict[str, str]) -> str:
    """Read and fingerprint an ordinary UTF-8 file inside the selected tree."""
    parts = _relative(relative).parts
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlink source is not admitted: {relative}")
    if not path.is_file():
        raise ValueError(f"missing source: {relative}")
    data = path.read_bytes()
    sources[relative] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8")


def _strings(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    """Validate declared identifier lists without manufacturing missing values."""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label}: expected a list of nonempty strings")
    if len(set(value)) != len(value) or (nonempty and not value):
        raise ValueError(f"{label}: duplicate or empty identifiers")
    return value


def _models(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep exact discovery records beside separately identified editorial text."""
    metadata = discovery.get("metadata")
    if (
        discovery.get("schema_version") != "2.0"
        or not isinstance(metadata, dict)
        or metadata.get("runtime") != "rust_native"
    ):
        raise ValueError("unsupported native discovery contract")
    models = discovery.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("native discovery has no model inventory")
    result = []
    seen = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("model discovery entries must be objects")
        key = model.get("key")
        if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
            raise ValueError("model key must be a canonical identifier")
        if key in seen:
            raise ValueError(f"duplicate model key: {key}")
        seen.add(key)
        if key not in MODEL_GUIDANCE:
            raise ValueError(f"new model needs editorial guidance: {key}")
        for field in ("family", "stability", "simulation_semantics"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ValueError(f"{key}: missing discovery field {field}")
        for field in ("supported_operations", "supported_backends", "simulation_engines", "variants"):
            _strings(model.get(field), f"{key}.{field}", nonempty=field != "variants")
        for field, value in model.items():
            if field.startswith("supports_") and not isinstance(value, bool):
                raise ValueError(f"{key}.{field}: capability flags must be boolean")
        title, summary = MODEL_GUIDANCE[key]
        result.append({"id": key, "title": title, "summary": summary, "advertised": model})
    return sorted(result, key=lambda model: model["id"])


def _frontmatter(text: str, relative: str) -> dict[str, str]:
    """Read the same bounded scalar metadata used by current tutorial contracts."""
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.DOTALL)
    if match is None:
        raise ValueError(f"{relative}: missing tutorial frontmatter")
    result = {}
    for key, raw in re.findall(r"^(title|description):[ \t]*(.*)$", match[1], re.MULTILINE):
        if key in result:
            raise ValueError(f"{relative}: duplicate frontmatter {key}")
        value = raw.strip()
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        if not isinstance(value, str) or not value or value[0] in "|>!&*[{":
            raise ValueError(f"{relative}: {key} requires a nonempty single-line scalar")
        result[key] = value
    if set(result) != {"title", "description"}:
        raise ValueError(f"{relative}: title and description are required")
    return result


def _route(relative: str) -> tuple[str, str]:
    """Map the canonical Markdown path to its existing trailing-slash route."""
    path = _relative(relative)
    if path.suffix not in {".md", ".mdx"}:
        raise ValueError(f"unsupported documentation source: {relative}")
    identifier = str(path.with_suffix(""))
    route = identifier.removesuffix("/index") if identifier != "index" else ""
    return identifier, "/innovate/" + (route + "/" if route else "")


def build_catalogue(root: Path) -> dict[str, Any]:
    """Derive source inventory and tutorial associations without runtime claims."""
    sources: dict[str, str] = {}
    _read(root, GENERATOR, sources)
    discovery = json.loads(_read(root, DISCOVERY, sources), object_pairs_hook=_object)
    contract = json.loads(_read(root, CONTENT, sources), object_pairs_hook=_object)
    if not isinstance(discovery, dict) or not isinstance(contract, dict):
        raise ValueError("source contracts must be JSON objects")
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        raise ValueError("unsupported content contract")
    pages = contract.get("pages")
    examples = contract.get("examples")
    if not isinstance(pages, dict) or not isinstance(examples, dict) or not examples:
        raise ValueError("content contract needs page and example inventories")
    model_records = _models(discovery)
    example_records = {}
    for path, declared in sorted(examples.items()):
        if not isinstance(declared, dict):
            raise ValueError(f"{path}: example declaration must be an object")
        _read(root, path, sources)
        if not isinstance(declared.get("status"), str) or not isinstance(declared.get("language"), str):
            raise ValueError(f"{path}: example status and language are required")
        example_records[path] = {
            "id": path,
            "source_sha256": sources[path],
            "declaration": declared,
            "tutorial_ids": [],
            "documented_routes": [],
        }
    tutorials = []
    seen_routes = set()
    for relative, declared in sorted(pages.items()):
        identifier, route = _route(relative)
        if not isinstance(declared, dict) or declared.get("kind") not in {
            "index",
            "reference",
            "tutorial",
            "historical",
        }:
            raise ValueError(f"{relative}: unknown page classification")
        if declared["kind"] == "historical":
            continue
        if route in seen_routes:
            raise ValueError(f"duplicate current documentation route: {route}")
        seen_routes.add(route)
        associated = _strings(declared.get("examples", []), f"{relative}.examples")
        if "example" in declared:
            associated = [declared["example"], *associated]
        associated = _strings(associated, f"{relative}.examples")
        for example in associated:
            if example not in example_records:
                raise ValueError(f"{relative}: unregistered example {example}")
            example_records[example]["documented_routes"].append(route)
        if declared["kind"] != "tutorial":
            continue
        if len(associated) != 1:
            raise ValueError(f"{relative}: current tutorials require one registered example")
        example = example_records[associated[0]]
        details = example["declaration"]
        if details["status"] != "runnable" or not isinstance(details.get("network"), bool):
            raise ValueError(f"{relative}: current tutorial example must be declared runnable with network policy")
        source = f"{PAGES}/{relative}"
        text = _read(root, source, sources)
        metadata = _frontmatter(text, relative)
        if re.search(r"(?m)^slug\s*:", text.split("---", 2)[1]):
            raise ValueError(f"{relative}: explicit slug needs a reviewed route mapping")
        example["tutorial_ids"].append(identifier)
        tutorials.append(
            {
                "id": identifier,
                "route": route,
                **metadata,
                "source": source,
                "example_id": example["id"],
                "language": details["language"],
                "network": details["network"],
                "execution_status": "not_run_by_generator",
            }
        )
    if not tutorials:
        raise ValueError("no current tutorials are registered")
    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "fingerprint_encoding": "sha256 of exact source-file bytes",
        "evidence_scope": {
            "source": "native discovery and registered current content",
            "native_execution": "not_run_by_generator",
            "scientific_acceptance": "not_established",
            "support_combinations": "not_inferred_from_independent_metadata_lists",
        },
        "discovery_metadata": discovery["metadata"],
        "models": model_records,
        "tutorials": tutorials,
        "examples": list(example_records.values()),
        "counts": {
            "models": len(model_records),
            "tutorials": len(tutorials),
            "tutorial_example_sources": len({tutorial["example_id"] for tutorial in tutorials}),
            "registered_examples": len(example_records),
        },
        "source_files": dict(sorted(sources.items())),
    }


def generate(root: Path, *, check: bool = False) -> bool:
    """Write the derived manifest, or report drift without modifying any file."""
    rendered = json.dumps(build_catalogue(root), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output = root / OUTPUT
    if output.is_symlink():
        raise ValueError("symlink catalogue output is not admitted")
    if check:
        return output.is_file() and output.read_bytes() == rendered.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered.encode("utf-8"))
    return True


def main(argv: list[str] | None = None) -> int:
    """Expose deterministic generation and a read-only freshness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        current = generate(arguments.root, check=arguments.check)
    except (ValueError, OSError, TypeError) as error:
        print(f"Learning catalogue: {error}", file=sys.stderr)
        return 1
    if not current:
        print("Learning catalogue is missing or stale; regenerate after reviewing source changes.", file=sys.stderr)
        return 1
    print("Learning catalogue: source inventory current; native execution not run by this command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
