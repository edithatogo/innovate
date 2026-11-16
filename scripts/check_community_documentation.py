"""Check documentation standards traceability, without granting external acceptance.

The reviewed source URLs and their access limitations are pinned. Repository
references are checked and fingerprinted, not treated as proof of their prose or
scientific claims. Registry identity and outcome evidence remain the responsibility
of the existing registry validator. Other named checks are recorded, never run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "docs/contracts/community-documentation.json"
PAGE = "docs/astro-site/src/content/docs/operations/documentation-standards.md"
REPOSITORY = "https://github.com/edithatogo/innovate"
TRACK = "conductor/tracks/research_software_registry_readiness_20260721"
REGISTRY_PATHS = {
    "manifest": f"{TRACK}/readiness.json",
    "checker": "scripts/check_registry_readiness.py",
    "policy": "docs/registry-readiness.md",
}
ISSUES = {"software_heritage": 399, "rrid": 400, "joss": 401}
HISTORY = "conductor/archive/community_submission_readiness_20260507/spec.md"
HISTORICAL_TARGETS = {
    "Apache Arrow",
    "PyPA",
    "pyOpenSci",
    "rOpenSci",
    "JOSS",
    "NumFOCUS",
    "scikit-learn-contrib",
    ".NET Foundation",
    "Julia/R community",
}
SOURCES = {
    "joss_review": ("JOSS review criteria", "https://joss.readthedocs.io/en/latest/review_criteria.html"),
    "joss_submission": ("JOSS submission guidance", "https://joss.readthedocs.io/en/latest/submitting.html"),
    "pyopensci_author": (
        "pyOpenSci author guide",
        "https://www.pyopensci.org/software-peer-review/how-to/author-guide.html",
    ),
    "pyopensci_policy": (
        "pyOpenSci policies",
        "https://www.pyopensci.org/software-peer-review/our-process/policies.html",
    ),
    "ropensci_packaging": ("rOpenSci packaging guide", "https://devguide.ropensci.org/pkg_building.html"),
    "ropensci_statistics": ("rOpenSci statistical standards", "https://stats-devguide.ropensci.org/standards.html"),
    "python_numpydoc": ("NumPy docstring guide", "https://numpydoc.readthedocs.io/en/latest/format.html"),
    "rust_documentation": (
        "Rust API documentation guidelines",
        "https://rust-lang.github.io/api-guidelines/documentation.html",
    ),
    "julia_documentation": ("Julia documentation manual", "https://docs.julialang.org/en/v1/manual/documentation/"),
    "software_heritage_archive": (
        "Software Heritage archival guidance",
        "https://www.softwareheritage.org/how-to-archive-reference-code/",
    ),
    "scicrunch_registry": (
        "SciCrunch curation guidance",
        "https://scicrunch.org/scicrunch/about/Curation%20Guidelines",
    ),
    "numfocus_projects": ("NumFOCUS project requirements", "https://numfocus.org/projects-overview"),
}
EXCERPT_SOURCES = {"software_heritage_archive", "scicrunch_registry"}
# Each entry pins identity, applicable sources and minimum evidence/check routes.
# It does not assert that those referenced checks passed or that prose is correct.
REQUIREMENTS = {
    "software_heritage_archival": (
        "Software Heritage archival",
        "software_heritage",
        "issue_commitment",
        {"software_heritage_archive"},
        {"docs/registry-readiness.md", f"{TRACK}/registry-evidence.md"},
        {"scripts/check_registry_readiness.py"},
    ),
    "rrid_identity": (
        "RRID resource identity",
        "rrid",
        "issue_commitment",
        {"scicrunch_registry"},
        {"docs/registry-readiness.md", f"{TRACK}/registry-evidence.md"},
        {"scripts/check_registry_readiness.py"},
    ),
    "joss_documentation": (
        "JOSS documentation",
        "joss",
        "issue_commitment",
        {"joss_review"},
        {
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "docs/astro-site/src/content/docs/user-guide/installation.md",
            "docs/contracts/api-documentation.json",
            "docs/contracts/content-quality.json",
        },
        {"scripts/check_api_documentation.py", "scripts/check_documentation_content.py"},
    ),
    "joss_submission_evidence": (
        "JOSS research and author evidence",
        "joss",
        "issue_commitment",
        {"joss_submission", "joss_review"},
        {"CITATION.cff", f"{TRACK}/joss-readiness.md", "docs/registry-readiness.md"},
        {"scripts/check_registry_readiness.py"},
    ),
    "python_review_documentation": (
        "pyOpenSci documentation guidance",
        "pyopensci",
        "adopted_guidance",
        {"pyopensci_author", "pyopensci_policy"},
        {
            "README.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "LICENSE",
            "docs/astro-site/src/content/docs/user-guide/getting-started.mdx",
            "docs/contracts/api-documentation.json",
        },
        {"scripts/check_api_documentation.py", "scripts/check_documentation_content.py"},
    ),
    "r_review_documentation": (
        "rOpenSci documentation guidance",
        "ropensci",
        "adopted_guidance",
        {"ropensci_packaging"},
        {
            "bindings/r/R/kernel_bridge.R",
            "bindings/r/man/innovate.R-package.Rd",
            "bindings/r/vignettes/innovate-r-kernel.Rmd",
            "docs/contracts/polyglot-api-documentation.json",
        },
        {"scripts/check_polyglot_api_documentation.py", "scripts/check_r_documentation.R"},
    ),
    "scientific_reporting": (
        "Scientific assumptions and reporting",
        "ropensci_statistics",
        "adopted_guidance",
        {"ropensci_statistics"},
        {
            "docs/astro-site/src/content/docs/native-migration.md",
            "docs/contracts/scientific-pedagogical-review.json",
            "migration-ledger.json",
        },
        {"scripts/check_scientific_review_contract.py"},
    ),
    "python_conventions": (
        "Python documentation conventions",
        "python",
        "adopted_guidance",
        {"python_numpydoc"},
        {"docs/contracts/api-documentation.json", "bindings/python/innovate/__init__.py"},
        {"scripts/check_api_documentation.py", "scripts/check_native_documentation.py"},
    ),
    "rust_conventions": (
        "Rust documentation conventions",
        "rust",
        "adopted_guidance",
        {"rust_documentation"},
        {"docs/contracts/api-documentation.json", "rust/crates/innovate-core/src/lib.rs"},
        {"scripts/check_api_documentation.py"},
    ),
    "julia_conventions": (
        "Julia documentation conventions",
        "julia",
        "adopted_guidance",
        {"julia_documentation"},
        {"bindings/julia/src/Innovate.jl", "docs/contracts/polyglot-api-documentation.json"},
        {"scripts/check_polyglot_api_documentation.py", "scripts/check_julia_documentation.jl"},
    ),
    "community_stewardship": (
        "Community stewardship guidance",
        "numfocus",
        "adopted_guidance",
        {"numfocus_projects"},
        {"CONTRIBUTING.md", "CODE_OF_CONDUCT.md", HISTORY},
        set(),
    ),
}


def _require(condition: object, message: str) -> None:
    """Reject invalid mappings with a specific, reviewable failure reason."""
    if not condition:
        raise ValueError(message)


def _fields(value: object, expected: set[str], context: str) -> None:
    """Reject missing and unrecognised fields, including unsupported status claims."""
    _require(isinstance(value, dict) and set(value) == expected, f"{context}: incorrect fields")


def _strings(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    """Require unambiguous nonempty strings with no duplicate entries."""
    _require(isinstance(value, list) and (bool(value) or allow_empty), f"{context}: a list is required")
    _require(
        all(isinstance(item, str) and item.strip() == item and item for item in value),
        f"{context}: nonempty strings are required",
    )
    _require(len(value) == len(set(value)), f"{context}: duplicate entries")
    return value


def _statement(value: object, context: str, *, minimum_words: int = 3) -> None:
    """Keep assessment text single-paragraph and free of hidden markup or links.

    This syntax check cannot establish the truth or quality of prose.
    """
    _require(isinstance(value, str) and len(value.split()) >= minimum_words, f"{context}: explanatory text required")
    _require(not any(character in value for character in "\n\r\x00<>[]"), f"{context}: use plain text")


def read_json(content: str | bytes) -> Any:
    """Read JSON while rejecting duplicate object keys rather than hiding drift."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique)


def _read(root: Path, relative: object) -> bytes:
    """Read nonempty repository evidence, rejecting ambiguous paths and escapes."""
    _require(
        isinstance(relative, str) and relative and "\\" not in relative and "\x00" not in relative,
        "Evidence path must be a canonical relative path",
    )
    path = Path(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts and str(path) == relative,
        f"Noncanonical evidence path: {relative}",
    )
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root.resolve()), f"Evidence path escapes the repository: {relative}")
    _require(resolved.is_file(), f"Missing evidence file: {relative}")
    content = resolved.read_bytes()
    _require(bool(content.strip()), f"Empty evidence file: {relative}")
    return content


def _digest(content: bytes) -> str:
    """Fingerprint the observed UTF-8/source bytes with LF newline normalization."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _registry_report(root: Path, manifest: object) -> dict:
    """Delegate outcomes to the existing repository policy without importing Innovate."""
    _require(
        _digest(_read(root, REGISTRY_PATHS["checker"])) == _digest(_read(ROOT, REGISTRY_PATHS["checker"])),
        "Mapped registry policy source differs from the trusted checker being executed",
    )
    spec = importlib.util.spec_from_file_location("innovate_registry_readiness", ROOT / REGISTRY_PATHS["checker"])
    _require(spec is not None and spec.loader is not None, "Registry policy is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_packet(root, manifest)


def validate(root: Path, packet: object) -> dict:
    """Validate a standards mapping and observe source hashes; do not certify software.

    Args:
        root: Repository containing the mapped documentation and registry evidence.
        packet: Parsed community-documentation contract to check.
    Returns:
        A traceability report with delegated registry integrity and pending work.
    Raises:
        ValueError: If identities, sources, evidence paths or claims are inconsistent.
        OSError: If an evidence file cannot be read.
    """
    root = root.resolve()
    _fields(
        packet,
        {
            "schema_version",
            "repository",
            "reviewed_on",
            "reviewed_by",
            "assessment",
            "registry",
            "sources",
            "requirements",
            "history",
        },
        "contract",
    )
    _require(type(packet["schema_version"]) is int and packet["schema_version"] == 1, "Unsupported schema_version")
    _require(packet["repository"] == REPOSITORY, "Repository identity mismatch")
    _require(packet["reviewed_by"] == "agent", "This mapping records agent review, not a human attestation")
    _require(packet["assessment"] == "documentation_traceability_only", "Unsupported assessment claim")
    reviewed = packet["reviewed_on"]
    _require(isinstance(reviewed, str), "Review date must be an ISO date")
    parsed_date = date.fromisoformat(reviewed)
    _require(str(parsed_date) == reviewed and parsed_date <= datetime.now(UTC).date(), "Invalid review date")

    registry = packet["registry"]
    _fields(registry, {"parent_issue", "issues", "paths"}, "registry")
    _require(
        type(registry["parent_issue"]) is int and registry["parent_issue"] == 398, "Registry parent issue must be #398"
    )
    _require(
        registry["issues"] == ISSUES and all(type(number) is int for number in registry["issues"].values()),
        "Registry child issue mapping changed",
    )
    _require(registry["paths"] == REGISTRY_PATHS, "Registry evidence paths must use the existing policy")
    observed = {path: _digest(_read(root, path)) for path in REGISTRY_PATHS.values()}
    registry_report = _registry_report(root, read_json(_read(root, REGISTRY_PATHS["manifest"])))

    sources = packet["sources"]
    _require(isinstance(sources, list), "Sources must be a list")
    source_ids = []
    for source in sources:
        _fields(source, {"id", "url", "checked_on", "access"}, "source")
        key = source["id"]
        _require(isinstance(key, str) and key in SOURCES, "Unknown official source ID")
        _require(source["url"] == SOURCES[key][1], f"{key}: changed official source URL requires review")
        _require(source["checked_on"] == reviewed, f"{key}: source review date mismatch")
        access = "official_search_excerpt" if key in EXCERPT_SOURCES else "official_page"
        _require(source["access"] == access, f"{key}: preserve the observed source access limitation")
        source_ids.append(key)
    _require(
        set(source_ids) == set(SOURCES) and len(source_ids) == len(SOURCES), "Missing or duplicate official source IDs"
    )

    requirements = packet["requirements"]
    _require(isinstance(requirements, list), "Requirements must be a list")
    requirement_ids = []
    checks = set()
    for entry in requirements:
        _fields(
            entry,
            {
                "id",
                "target",
                "relationship",
                "source_ids",
                "section",
                "requirement",
                "status",
                "evidence",
                "checks",
                "remaining",
            },
            "requirement",
        )
        key = entry["id"]
        _require(isinstance(key, str) and key in REQUIREMENTS, "Unknown requirement ID")
        _, target, relationship, required_sources, evidence, required_checks = REQUIREMENTS[key]
        _require(
            entry["target"] == target and entry["relationship"] == relationship,
            f"{key}: target or commitment relationship changed",
        )
        _require(entry["status"] == "mapped_not_qualified", f"{key}: mapping is not an acceptance status")
        _require(
            set(_strings(entry["source_ids"], f"{key} sources")) == required_sources,
            f"{key}: incorrect applicable official sources",
        )
        _statement(entry["section"], f"{key} section", minimum_words=1)
        _statement(entry["requirement"], f"{key} requirement")
        for remaining in _strings(entry["remaining"], f"{key} remaining work"):
            _statement(remaining, f"{key} remaining work")
        paths = _strings(entry["evidence"], f"{key} evidence")
        _require(evidence <= set(paths), f"{key}: required evidence references removed")
        scripts = _strings(entry["checks"], f"{key} checks", allow_empty=True)
        _require(set(scripts) == required_checks, f"{key}: delegated check routes changed")
        for path in [*paths, *scripts]:
            observed[path] = _digest(_read(root, path))
        checks.update(scripts)
        requirement_ids.append(key)
    _require(
        set(requirement_ids) == set(REQUIREMENTS) and len(requirement_ids) == len(REQUIREMENTS),
        "Missing or duplicate requirement IDs",
    )

    history = packet["history"]
    _fields(history, {"path", "status", "targets"}, "historical provenance")
    _require(
        history["path"] == HISTORY and history["status"] == "requires_reassessment",
        "Keep historical aspirations separate from current commitments",
    )
    targets = _strings(history["targets"], "historical targets")
    _require(set(targets) == HISTORICAL_TARGETS, "Historical target inventory changed")
    historical_text = _read(root, HISTORY)
    _require(all(target.encode() in historical_text for target in targets), "Historical target lacks source provenance")
    observed[HISTORY] = _digest(historical_text)

    return {
        "schema_version": 1,
        "record_valid": True,
        "assessment": "documentation_traceability_only",
        "reviewed_on": reviewed,
        "online_verification_performed": False,
        "native_execution_performed": False,
        "external_acceptance_assessed": False,
        "scientific_validation_assessed": False,
        "release_qualification_assessed": False,
        "requirements_mapped": requirement_ids,
        "requirements_with_pending_work": requirement_ids,
        "checks_not_run": sorted(checks - {REGISTRY_PATHS["checker"]}),
        "registry": registry_report,
        "mapping_sha256": _digest(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()),
        "observed_source_sha256": dict(sorted(observed.items())),
    }


def _repository_link(path: str) -> str:
    """Link to the mapped repository source with an unambiguous path label."""
    return f"[{path}]({REPOSITORY}/blob/main/{path})"


def render_page(packet: dict, report: dict) -> str:
    """Project the validated mapping and recorded registry states into current guidance.

    Run validate first. The generated page deliberately reports remaining work
    and cannot add a free-form acceptance badge or unrecorded provider outcome.
    """
    lines = [
        "---",
        "title: Documentation standards and research communities",
        "description: Current standards sources, mapped documentation and remaining research-software evidence.",
        "---",
        "",
        "Innovate uses research-software review guidance to make its documentation useful,",
        "reproducible and clear about scientific limits. This page maps that guidance to",
        "current repository evidence. It records neither community approval nor complete",
        "conformance to every standard.",
        "",
        f"The official guidance was reviewed by an agent on **{packet['reviewed_on']}**.",
        "That dated assessment is not a human author declaration or a live provider check.",
        "Recheck official policies before any application or submission.",
        "",
        "## Current issue commitments",
        "",
        f"[Issue #398]({REPOSITORY}/issues/398) owns three registry deliverables.",
        "The following states come from the existing readiness record; its validator",
        "checks retained evidence integrity without authenticating provider responses.",
        "",
        "| Destination | Owning issue | Recorded state |",
        "| --- | --- | --- |",
    ]
    labels = {"software_heritage": "Software Heritage", "rrid": "RRID", "joss": "JOSS"}
    for destination in report["registry"]["destinations"]:
        key, issue, state = destination["key"], destination["issue"], destination["state"]
        lines.append(f"| {labels[key]} | [#{issue}]({REPOSITORY}/issues/{issue}) | `{state}` |")
    lines.extend(
        [
            "",
            "Follow the [registry readiness policy](/innovate/operations/registry-readiness/)",
            "for selected-release identity, provider receipts, author evidence and unresolved",
            "work. Scientific qualification and release acceptance remain separate.",
            "",
            f"[Issue #556]({REPOSITORY}/issues/556) and [#559]({REPOSITORY}/issues/559)",
            "concern scientific ecosystem and consumer interoperability qualification.",
            "They do not establish admission to an external review community.",
            "",
            "## Standards mapped to documentation",
            "",
            "Each entry below is **mapped, not qualified**. Evidence links identify material",
            "to inspect; named check scripts are routes to run, not passing results. Python,",
            "Julia and R retain their native documentation conventions, with Rust as the",
            "scientific core. Stricter local rules remain Innovate policy.",
            "",
        ]
    )
    for entry in packet["requirements"]:
        label = REQUIREMENTS[entry["id"]][0]
        sources = ", ".join(f"[{SOURCES[key][0]}]({SOURCES[key][1]})" for key in entry["source_ids"])
        relation = "Current issue commitment" if entry["relationship"] == "issue_commitment" else "Adopted guidance"
        lines.extend(
            [
                f"### {label}",
                "",
                entry["requirement"],
                "",
                f"**{relation}.** Source: {sources}. Applicable section: {entry['section']}.",
                "",
                "Mapped evidence: " + "; ".join(_repository_link(path) for path in entry["evidence"]) + ".",
                "",
            ]
        )
        if entry["checks"]:
            lines.extend(["Check routes: " + "; ".join(_repository_link(path) for path in entry["checks"]) + ".", ""])
        lines.extend(["Still required:", ""])
        lines.extend(f"- {remaining}" for remaining in entry["remaining"])
        lines.append("")
    lines.extend(
        [
            "## Historical community aspirations",
            "",
            "The "
            + _repository_link(HISTORY)
            + " records earlier targets: "
            + ", ".join(packet["history"]["targets"])
            + ".",
            "",
            "Those plans require reassessment against the current Rust/Polars implementation.",
            "Historical completion labels and readiness flags remain evidence of their",
            "original scope. They do not grant current admission, registration, affiliation",
            "or review acceptance. Removing obsolete site pages does not remove that source",
            "history or its outstanding obligations.",
            "",
            "## Source access and limits",
            "",
            "The Software Heritage and SciCrunch guidance was available through indexed",
            "official-page excerpts during this review. Direct access timed out or was",
            "blocked; no live archive entry, duplicate-resource search or curator decision",
            "was verified. The other linked guidance pages were read directly.",
            "",
            "The selected rOpenSci statistical rules are an applicability map, not a claim",
            "to satisfy the full statistical review programme. Documentation presence,",
            "executed examples, scientific review, empirical validity and external outcomes",
            "remain distinct. A public-data fit does not demonstrate causal identification.",
            "",
            "## Checking and maintaining this mapping",
            "",
            "The " + _repository_link(CONTRACT) + " is the source for this page.",
            "Use the selected Python 3.15.2 environment from the repository's contributor",
            "instructions:",
            "",
            "```sh",
            ".venv-native/bin/python scripts/check_community_documentation.py --json",
            "```",
            "",
            "The check validates official source identities, review dates, issue ownership,",
            "required repository references and this page's exact projection. It delegates",
            "registry evidence to the existing readiness checker and reports source hashes.",
            "It does not execute the listed language/example checks, contact providers,",
            "submit an application or establish scientific or release acceptance.",
            "",
            "After reviewing a change to the mapping or registry record, regenerate this",
            "page with `--write-page`, then run the normal check. Regeneration updates the",
            "projection; it supplies no new human, runtime or provider evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def check(root: Path, manifest: str = CONTRACT, *, write_page: bool = False) -> dict:
    """Validate the record and check or explicitly regenerate its sole site projection."""
    packet = read_json(_read(root, manifest))
    report = validate(root, packet)
    rendered = render_page(packet, report)
    if write_page:
        destination = root / PAGE
        _require(destination.resolve().is_relative_to(root.resolve()), "Standards page escapes the repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    actual = _read(root, PAGE).decode("utf-8").replace("\r\n", "\n")
    _require(actual == rendered, "Standards page differs from the mapping; review changes before --write-page")
    report["page_matches_mapping"] = True
    report["observed_source_sha256"][PAGE] = _digest(actual.encode())
    return report


def main(argv: list[str] | None = None) -> int:
    """Return zero for valid traceability and two for invalid evidence or page drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root containing the evidence")
    parser.add_argument("--manifest", default=CONTRACT, help="contract path relative to the repository")
    parser.add_argument("--write-page", action="store_true", help="regenerate the page after reviewing source changes")
    parser.add_argument("--json", action="store_true", help="print the scoped machine-readable report")
    args = parser.parse_args(argv)
    try:
        report = check(args.root, args.manifest, write_page=args.write_page)
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        report = {"record_valid": False, "online_verification_performed": False, "error": str(exc)}
        code = 2
    else:
        code = 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif code:
        print(f"Invalid documentation standards mapping: {report['error']}")
    else:
        print(f"Documentation traceability valid: {len(report['requirements_mapped'])} mapped requirements.")
        print("Runtime execution, scientific qualification and external acceptance remain separate.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
