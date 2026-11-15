"""Validate scoped review consistency and freshness of its declared source inputs.

This standard-library gate checks a review record, not scientific validity or
pedagogical quality. It performs no native execution, browser work or human
reader trial. Schema 1 records agent review with unexecuted human tasks only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("docs/contracts/scientific-pedagogical-review.json")
SITE = PurePosixPath("docs/astro-site/src/content/docs")
ENCODING = "sha256 of UTF-8 bytes with CRLF normalized to LF"
COMPLETION = {
    "scoped_agent_review": True,
    "human_reader_validation": False,
    "all_current_pages_reviewed": False,
    "all_historical_pages_reviewed": False,
    "scientific_model_validation": False,
    "complete_documentation_quality_qualification": False,
}
REQUIRED = {
    "schema_version",
    "review_date",
    "review_timezone",
    "base_commit",
    "reviewer",
    "status",
    "scope",
    "fingerprint_encoding",
    "source_files",
    "references",
    "findings",
    "reader_tasks",
    "verification",
    "completion",
    "remaining",
}
OPTIONAL = {"recovery", "independent_reviews"}


def _text(value: object) -> bool:
    """Return whether a value is a nonempty, non-whitespace string."""
    return isinstance(value, str) and bool(value.strip())


def _strings(value: object) -> bool:
    """Require a nonempty list of distinct, nonempty strings."""
    return (
        isinstance(value, list) and bool(value) and all(_text(item) for item in value) and len(value) == len(set(value))
    )


def _relative(value: object) -> bool:
    """Recognize a canonical repository-relative POSIX path."""
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and "\0" not in value
        and not PurePosixPath(value).is_absolute()
        and ".." not in PurePosixPath(value).parts
        and PurePosixPath(value).as_posix() == value
        and value != "."
    )


def _source(root: Path, value: object) -> Path:
    """Resolve an existing source file without leaving the repository."""
    if not _relative(value):
        raise ValueError(f"Invalid source path: {value!r}.")
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Source is absent or outside the repository: {value}.")
    return path


def _digest(path: Path) -> str:
    """Hash UTF-8 source bytes after the record's explicit CRLF normalization."""
    data = path.read_bytes()
    data.decode("utf-8")
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects rather than accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _records(value: object, label: str, errors: list[str]) -> list[dict[str, Any]]:
    """Collect dictionary rows while rejecting empty or duplicate record IDs."""
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a nonempty list.")
        return []
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            errors.append(f"{label} must contain objects.")
            continue
        identifier = row.get("id")
        if not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9-]*", identifier) is None:
            errors.append(f"{label} has an invalid id.")
        elif identifier in identifiers:
            errors.append(f"{label} has duplicate id {identifier!r}.")
        else:
            identifiers.add(identifier)
        rows.append(row)
    return rows


def validate_review(root: Path, record: object) -> list[str]:
    """Return consistency and freshness errors for one scoped agent review.

    Args:
        root: Repository root containing every declared source.
        record: Decoded JSON review record; malformed structures are rejected.

    Returns:
        Human-readable errors. An empty list establishes only record consistency
        and source freshness within the declared review scope.
    """
    root = root.resolve()
    if not isinstance(record, dict) or not REQUIRED <= set(record):
        return ["Review record is missing required schema-1 sections."]
    errors: list[str] = []
    if not set(record) <= REQUIRED | OPTIONAL:
        errors.append("Review record contains unsupported top-level sections.")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        errors.append("schema_version must be integer 1.")
    if record["status"] != "scoped_agent_review_complete":
        errors.append("status must describe the completed scoped agent review.")
    value = record["review_date"]
    try:
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError("date format")
        date.fromisoformat(value)
    except ValueError:
        errors.append("review_date must be a valid ISO calendar date.")
    if not _text(record["review_timezone"]):
        errors.append("review_timezone must be recorded.")
    if not isinstance(record["base_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", record["base_commit"]) is None:
        errors.append("base_commit must be a full lowercase Git commit SHA.")
    reviewer = record["reviewer"]
    if (
        not isinstance(reviewer, dict)
        or set(reviewer) != {"kind", "role", "human_reviewer"}
        or reviewer.get("kind") != "agent"
        or reviewer.get("human_reviewer") is not False
        or not _text(reviewer.get("role"))
    ):
        errors.append("Schema 1 requires an explicitly identified agent reviewer.")
    if record["fingerprint_encoding"] != ENCODING:
        errors.append("fingerprint_encoding differs from the checked UTF-8/CRLF contract.")

    scope = record["scope"]
    if not isinstance(scope, dict) or set(scope) != {"pages", "native_guide_sections", "focus", "exclusions"}:
        return errors + ["scope must declare pages, native_guide_sections, focus and exclusions."]
    for key in ("pages", "native_guide_sections", "focus", "exclusions"):
        if not _strings(scope[key]):
            errors.append(f"scope.{key} must be a nonempty list of distinct strings.")
    if not _strings(scope["pages"]):
        return errors
    pages = set(scope["pages"])
    for page in pages:
        if (
            not _relative(page)
            or not PurePosixPath(page).is_relative_to(SITE)
            or PurePosixPath(page).suffix not in {".md", ".mdx"}
        ):
            errors.append(f"Reviewed page is not a current documentation source: {page!r}.")

    sources = record["source_files"]
    if not isinstance(sources, dict) or not sources:
        return errors + ["source_files must contain reviewed source fingerprints."]
    if not pages <= set(sources):
        errors.append("Every reviewed page must have a source fingerprint.")
    source_texts: dict[str, str] = {}
    for relative, expected in sources.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            errors.append(f"Invalid SHA-256 fingerprint: {relative!r}.")
            continue
        try:
            path = _source(root, relative)
            if _digest(path) != expected:
                errors.append(f"Reviewed source changed: {relative}. Re-review before updating its fingerprint.")
            source_texts[relative] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for page in pages:
        for imported in re.findall(
            r"^import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", source_texts.get(page, ""), re.MULTILINE
        ):
            if not imported.startswith(".") or not (imported.endswith("?raw") or imported.endswith(".json")):
                continue
            path = (root / page).parent.joinpath(imported.removesuffix("?raw")).resolve()
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in sources:
                errors.append(f"Reviewed page import has no source fingerprint: {page} -> {imported}.")

    references = _records(record["references"], "references", errors)
    reference_ids = {row["id"] for row in references if isinstance(row.get("id"), str)}
    page_text = "\n".join(source_texts.get(page, "") for page in pages)
    for row in references:
        if not all(_text(row.get(key)) for key in ("title", "source_type", "supports", "limits")):
            errors.append(f"Reference {row.get('id')!r} needs bibliographic and claim-boundary text.")
        if not _strings(row.get("inspected_locations")):
            errors.append(f"Reference {row.get('id')!r} needs inspected locations.")
        url = row.get("url")
        if not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
            errors.append(f"Reference {row.get('id')!r} needs an HTTPS source URL.")
        elif url not in page_text:
            errors.append(f"Reference URL is absent from reviewed pages: {row.get('id')!r}.")

    findings = _records(record["findings"], "findings", errors)
    for row in findings:
        if not isinstance(row.get("page"), str) or row["page"] not in pages or row.get("status") != "resolved_in_prose":
            errors.append(f"Finding {row.get('id')!r} must identify a reviewed page and resolved status.")
        if (
            not isinstance(row.get("severity"), str)
            or row["severity"] not in {"low", "medium", "high"}
            or not all(_text(row.get(key)) for key in ("finding", "resolution"))
        ):
            errors.append(f"Finding {row.get('id')!r} lacks severity, finding or resolution.")
        refs = row.get("references")
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in reference_ids for ref in refs):
            errors.append(f"Finding {row.get('id')!r} has an unknown reference.")

    tasks = _records(record["reader_tasks"], "reader_tasks", errors)
    task_pages: set[str] = set()
    for row in tasks:
        if not isinstance(row.get("page"), str) or row["page"] not in pages:
            errors.append(f"Reader task {row.get('id')!r} is outside the reviewed scope.")
        else:
            task_pages.add(row["page"])
        if (
            not all(_text(row.get(key)) for key in ("prompt", "expected_answer"))
            or row.get("answer_review") != "agent_checked"
        ):
            errors.append(f"Reader task {row.get('id')!r} needs a checked expected answer.")
        if row.get("human_execution") != "not_run":
            errors.append("Schema 1 has no human trial evidence; human_execution must remain not_run.")
    if task_pages != pages:
        errors.append("Every reviewed page needs at least one recorded reader task.")

    verification = record["verification"]
    if not isinstance(verification, dict):
        errors.append("verification must explicitly record test and human-review limits.")
    else:
        for key in (
            "python_fences",
            "raw_example_imports",
            "independent_arithmetic",
            "native_execution",
            "astro_build",
            "citation_review",
        ):
            if not _text(verification.get(key)):
                errors.append(f"verification.{key} must state the evidence or limitation.")
        for key in ("human_participants", "human_task_trials"):
            if type(verification.get(key)) is not int or verification[key] != 0:
                errors.append(f"Schema 1 has no human trial evidence; verification.{key} must be integer zero.")
    completion = record["completion"]
    if not isinstance(completion, dict) or set(completion) != set(COMPLETION):
        errors.append("completion must contain exactly the scoped qualification flags.")
    else:
        for key, expected in COMPLETION.items():
            if completion[key] is not expected:
                errors.append(f"completion.{key} must remain {str(expected).lower()} for this scoped agent review.")
    if not _strings(record["remaining"]):
        errors.append("remaining must retain explicit unqualified work.")

    independent = record.get("independent_reviews")
    if independent is not None:
        if not isinstance(independent, dict) or set(independent) != {
            "source_commit",
            "reviewer_kind",
            "independent_of_page_author",
            "results",
            "limits",
        }:
            errors.append("independent_reviews has an unsupported evidence structure.")
        else:
            if (
                not isinstance(independent["source_commit"], str)
                or re.fullmatch(r"[0-9a-f]{40}", independent["source_commit"]) is None
            ):
                errors.append("Independent review needs an immutable source commit.")
            if independent["reviewer_kind"] != "agent" or independent["independent_of_page_author"] is not True:
                errors.append("Independent review must identify its agent and author relationship.")
            if not _text(independent["limits"]):
                errors.append("Independent review must state its limits.")
            results = independent["results"]
            seen: set[str] = set()
            if not isinstance(results, list) or not results:
                errors.append("Independent review results must be nonempty.")
            else:
                for row in results:
                    if not isinstance(row, dict):
                        errors.append("Independent review results must be objects.")
                        continue
                    page = row.get("page")
                    if not isinstance(page, str) or page not in pages or page in seen:
                        errors.append("Independent review results need unique scoped page paths.")
                    else:
                        seen.add(page)
                    if row.get("outcome") != "no_blocking_findings_in_scoped_read" or not _strings(row.get("checks")):
                        errors.append(f"Independent review of {page!r} lacks bounded checks/outcome.")
                    inputs = row.get("source_paths")
                    if not _strings(inputs) or any(path not in sources for path in inputs):
                        errors.append(f"Independent review of {page!r} names untracked source inputs.")
                    refs = row.get("references")
                    if not isinstance(refs, list) or any(
                        not isinstance(ref, str) or ref not in reference_ids for ref in refs
                    ):
                        errors.append(f"Independent review of {page!r} has an unknown reference.")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Write a bounded JSON consistency report and return zero only on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report here as well as to stdout; relative paths use the invocation directory.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    record: object = None
    try:
        path = _source(root, CONTRACT.as_posix())
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        errors = validate_review(root, record)
    except (OSError, ValueError) as exc:
        errors = [f"Review record could not be validated: {exc}"]
    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "scope": "review_record_consistency_and_declared_source_freshness",
        "errors": errors,
        "source_files_declared": len(record.get("source_files", {}))
        if isinstance(record, dict) and isinstance(record.get("source_files"), dict)
        else 0,
        "scientific_quality_certified": False,
        "pedagogical_quality_certified": False,
        "human_reader_validation_performed": False,
        "native_execution_performed": False,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
