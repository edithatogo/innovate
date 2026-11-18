"""Check the retained JOSS draft offline without granting submission readiness.

This is a source/edition drift and bounded Markdown structure check. Actual YAML,
BibTeX and PDF validation belongs to the separately recorded rendering step.
Hashes identify bytes; they do not authenticate a commit or establish research
use, scientific validity, author consent or provider/editorial acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK = "conductor/tracks/research_software_registry_readiness_20260721"
SNAPSHOT = f"{TRACK}/joss/source-snapshot.json"
PAPER = f"{TRACK}/joss/paper.md"
CLAIMS = f"{TRACK}/joss/claim-evidence.md"
SOURCE_INPUTS = {
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    "migration-ledger.json",
    "conductor/product.md",
    "conductor/release-gates-20261107.md",
    "docs/astro-site/src/content/docs/native-migration.md",
    "rust/crates/innovate-core/Cargo.toml",
    "rust/crates/innovate-tabular/Cargo.toml",
    "rust/crates/innovate-tabular/src/lib.rs",
    ".github/workflows/ci.yml",
}
EDITION_INPUTS = {PAPER, CLAIMS, f"{TRACK}/joss/paper.bib"}
SECTIONS = (
    "Summary",
    "Statement of need",
    "State of the field",
    "Software design",
    "Research impact statement",
    "AI usage disclosure",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_snapshot(path: Path) -> object:
    """Reject duplicate fields instead of trusting whichever value appears last."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def verified_text(root: Path, name: str, digest: object) -> tuple[str, bool]:
    """Read only an allowlisted text input, with explicit Git CRLF handling."""
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Input must stay within repository: {name}")
    require(path.is_file(), f"Missing input: {name}")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"Invalid SHA-256: {name}")
    raw = path.read_bytes()
    require(raw.strip(), f"Empty input: {name}")
    normalized = hashlib.sha256(raw).hexdigest() != digest
    if normalized:
        require(b"\r\n" in raw, f"Input digest mismatch: {name}")
        raw = raw.replace(b"\r\n", b"\n")
        require(hashlib.sha256(raw).hexdigest() == digest, f"Input digest mismatch: {name}")
    return raw.decode("utf-8"), normalized


def visible_lines(body: str) -> list[str]:
    """Exclude comments and fenced code from this edition's ATX section check."""
    lines = []
    fence = None
    in_comment = False
    for line in body.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker and not in_comment:
            fence = marker[1]
            continue
        visible = ""
        while line:
            if in_comment:
                end = line.find("-->")
                if end < 0:
                    break
                line, in_comment = line[end + 3 :], False
            else:
                start = line.find("<!--")
                if start < 0:
                    visible += line
                    break
                visible += line[:start]
                line, in_comment = line[start + 4 :], True
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", visible)
        if marker:
            fence = marker[1]
        else:
            lines.append(visible)
    require(not in_comment, "Unclosed Markdown HTML comment")
    require(fence is None, "Unclosed Markdown code fence")
    return lines


def manuscript_structure(paper: str) -> int:
    """Validate required headings and a reproducible, non-editorial word count."""
    metadata = re.match(r"\A---\r?\n.+?\r?\n---(?:\r?\n|$)", paper, re.DOTALL)
    require(metadata is not None, "Manuscript requires YAML front matter; render separately to parse it")
    lines = visible_lines(paper[metadata.end() :])
    headings = [(index, line[2:].strip()) for index, line in enumerate(lines) if line.startswith("# ")]
    positions = []
    for name in SECTIONS:
        matches = [index for index, heading in headings if heading == name]
        require(len(matches) == 1, f"Required section must occur exactly once: {name}")
        positions.append(matches[0])
    require(positions == sorted(positions), "Required section order differs from the reviewed edition")
    references = [index for index, heading in headings if heading == "References"]
    require(len(references) == 1 and references[0] > positions[-1], "One final References heading is required")
    require(headings[-1][1] == "References", "References must be the final section")
    for index, name in zip(positions, SECTIONS, strict=True):
        end = next(offset for offset, _ in headings if offset > index)
        prose = "\n".join(lines[index + 1 : end]).strip()
        require(bool(prose), f"Required section is empty: {name}")
    body = "\n".join(line for line in lines[positions[0] : references[0]] if not line.startswith("#"))
    body = re.sub(r"\[@[^\]]+\]", "", body)
    words = len(re.findall(r"\b\w+(?:[-’']\w+)*\b", body))
    require(750 <= words <= 1750, f"Draft prose word count {words} is outside the recorded 750–1750 guidance")
    return words


def validate_packet(root: Path, packet: object) -> dict:
    """Validate the current unapproved-draft schema; never refresh its hashes."""
    require(isinstance(packet, dict), "Snapshot must be an object")
    require(type(packet.get("schema_version")) is int and packet["schema_version"] == 1, "Unsupported schema_version")
    require(packet.get("state") == "unapproved_working_draft", "This checker accepts only an unapproved working draft")
    require(packet.get("submission_ready") is False, "submission_ready must remain false for this draft schema")
    revision = packet.get("software_revision")
    require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) and set(revision) != {"0"},
        "A full, non-placeholder software revision is required",
    )
    inputs = {}
    normalized = []
    for group, expected in (("source_sha256", SOURCE_INPUTS), ("edition_sha256", EDITION_INPUTS)):
        fingerprints = packet.get(group)
        require(
            isinstance(fingerprints, dict) and set(fingerprints) == expected, f"{group}: exact reviewed inputs required"
        )
        for name, digest in fingerprints.items():
            inputs[name], changed = verified_text(root, name, digest)
            if changed:
                normalized.append(name)
    project = tomllib.loads(inputs["pyproject.toml"]).get("project")
    require(isinstance(project, dict), "pyproject.toml project must be a table")
    version = project.get("version")
    require(
        isinstance(version, str) and bool(version.strip()) and version == packet.get("software_version"),
        "Software version is empty or differs from project",
    )
    for name in (PAPER, CLAIMS):
        visible = "\n".join(visible_lines(inputs[name]))
        require(re.search(rf"(?<!\w){revision}(?!\w)", visible), f"Recorded software revision is absent from {name}")
        require(
            re.search(rf"(?<![\w.+-]){re.escape(version)}(?![\w.+-])", visible),
            f"Recorded software version is absent from {name}",
        )
    require(
        any(
            line.startswith("> Working draft: not author-approved or submission-ready.")
            for line in visible_lines(inputs[PAPER])
        ),
        "The visible unapproved draft notice is required",
    )
    words = manuscript_structure(inputs[PAPER])
    return {
        "snapshot_valid": True,
        "state": packet["state"],
        "software_revision": revision,
        "software_version": version,
        "source_files_checked": len(SOURCE_INPUTS),
        "edition_files_checked": len(EDITION_INPUTS),
        "normalized_text_paths": sorted(normalized),
        "required_sections_checked": len(SECTIONS),
        "prose_word_count": words,
        "submission_ready": False,
        "git_revision_verified": False,
        "scientific_qualification_performed": False,
        "online_verification_performed": False,
        "limits": (
            "Structural and byte-identity checks only; rendering, research use "
            "and author/editorial decisions are separate."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository checkout to check (no network access)")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument(
        "--require-submission-ready", action="store_true", help="Fail while this draft remains unapproved"
    )
    args = parser.parse_args(argv)
    try:
        report = validate_packet(args.root, read_snapshot(args.root / SNAPSHOT))
    except (OSError, ValueError) as error:
        report = {"snapshot_valid": False, "submission_ready": False, "error": str(error)}
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["snapshot_valid"]:
        print(f"Draft integrity valid: {report['prose_word_count']} prose words; submission readiness remains false.")
    else:
        print(f"Draft integrity failed: {report['error']}")
    return int(not report["snapshot_valid"] or (args.require_submission_ready and not report["submission_ready"]))


if __name__ == "__main__":
    raise SystemExit(main())
