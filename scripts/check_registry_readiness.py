"""Validate retained registry evidence offline; never submit or verify online."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TRACK = "conductor/tracks/research_software_registry_readiness_20260721"
MANIFEST = f"{TRACK}/readiness.json"
REPOSITORY = "https://github.com/edithatogo/innovate"
INPUTS = {"CITATION.cff", "LICENSE", "pyproject.toml", "migration-ledger.json"}
ISSUES = {"software_heritage": 399, "rrid": 400, "joss": 401}
HOSTS = {
    "software_heritage": {"archive.softwareheritage.org"},
    "rrid": {"scicrunch.org", "rrid.site", "identifiers.org"},
    "joss": {"joss.theoj.org", "doi.org", "github.com"},
}
IDENTIFIERS = {
    "software_heritage": r"swh:1:(?:rev|rel):[0-9a-f]{40}",
    "rrid": r"RRID:SCR_[0-9]{6,}",
    "joss": r"10\.21105/joss\.[0-9]{5,}",
}


def require(condition: object, message: str) -> None:
    """Reject an incomplete or inconsistent record with an actionable reason."""
    if not condition:
        raise ValueError(message)


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def hex_digest(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None and set(value) != {"0"}


def read_json(content: bytes) -> object:
    """Reject ambiguous JSON objects instead of accepting their last duplicate key."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique_object)


def read_local(root: Path, name: object, digest: object = None) -> bytes:
    """Read a nonempty repository file, rejecting traversal and stale evidence."""
    require(nonempty(name), "Evidence path must be nonempty")
    path = Path(name)
    resolved = (root / path).resolve()
    require(not path.is_absolute() and resolved.is_relative_to(root.resolve()), "Evidence must stay within repository")
    require(resolved.is_file(), f"Missing evidence file: {name}")
    content = resolved.read_bytes()
    require(content.strip(), f"Empty evidence file: {name}")
    if digest is not None:
        require(hex_digest(digest, 64), f"Invalid SHA-256 for {name}")
        matches_raw = hashlib.sha256(content).hexdigest() == digest
        # Windows git checkouts with core.autocrlf=true convert LF to CRLF on text evidence files.
        matches_lf = b"\r\n" in content and hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() == digest
        require(matches_raw or matches_lf, f"Evidence digest mismatch: {name}")
    return content


def validate_evidence(root: Path, packet: dict, entry: dict) -> None:
    """Check the integrity and identity of a recorded provider event, not its truth."""
    key, state = entry["key"], entry["state"]
    evidence = entry.get("evidence")
    require(isinstance(evidence, dict), f"{key}: provider receipt required for {state}")
    require(set(evidence) == {"path", "sha256"}, f"{key}: receipt path and digest required")
    require(hex_digest(evidence["sha256"], 64), f"{key}: receipt digest required")
    receipt = read_json(read_local(root, evidence["path"], evidence["sha256"]))
    require(isinstance(receipt, dict), f"{key}: provider receipt must be an object")
    for field, expected in (
        ("provider", key),
        ("repository", REPOSITORY),
        ("software_version", packet["software_version"]),
        ("event", state),
        ("identifier", entry.get("identifier")),
    ):
        require(field in receipt and receipt[field] == expected, f"{key}: receipt {field} mismatch")
    require(nonempty(receipt.get("summary")), f"{key}: explain the provider outcome")
    require(nonempty(receipt.get("checked_at")), f"{key}: checked_at required")
    checked = datetime.fromisoformat(receipt["checked_at"])
    require(checked.tzinfo is not None, f"{key}: checked_at must include a timezone")
    require(nonempty(receipt.get("url")), f"{key}: authoritative provider URL required")
    url = urlsplit(receipt["url"])
    require(
        url.scheme == "https"
        and url.hostname in HOSTS[key]
        and url.username is None
        and url.password is None
        and url.port in {None, 443},
        f"{key}: untrusted provider URL",
    )
    if url.hostname == "github.com":
        require(
            url.path.startswith("/openjournals/joss-reviews/issues/"), "JOSS receipt must identify its review issue"
        )
    status = receipt.get("http_status")
    require(type(status) is int and 200 <= status < 300, f"{key}: a successfully retrieved provider event is required")
    require(hex_digest(receipt.get("response_sha256"), 64), f"{key}: retained response digest required")
    response = read_local(root, receipt.get("response_path"), receipt["response_sha256"])
    identifier = entry.get("identifier")
    if state != "verified":
        require(identifier is None, f"{key}: only a verified record can advertise an identifier")
        return
    require(isinstance(identifier, str), f"{key}: verified record needs an identifier")
    require(re.fullmatch(IDENTIFIERS[key], identifier), f"{key}: invalid identifier")
    suffix = identifier.rsplit(":" if key == "software_heritage" else "_" if key == "rrid" else ".", 1)[1]
    require(set(suffix) != {"0"}, f"{key}: placeholder identifier")
    resolver_paths = {
        "software_heritage": {("archive.softwareheritage.org", f"/api/1/resolve/{identifier}")},
        "rrid": {("scicrunch.org", f"/resolver/{identifier}")},
        "joss": {("joss.theoj.org", f"/papers/{identifier}"), ("doi.org", f"/{identifier}")},
    }
    require(
        not url.query and not url.fragment and (url.hostname, unquote(url.path).rstrip("/")) in resolver_paths[key],
        f"{key}: URL must identify the supported provider resolver path",
    )
    if key == "software_heritage":
        resolved = read_json(response)
        require(isinstance(resolved, dict), "software_heritage: resolver response must be an object")
        object_kind = "revision" if identifier.startswith("swh:1:rev:") else "release"
        require(
            resolved.get("namespace") == "swh"
            and type(resolved.get("scheme_version")) is int
            and resolved["scheme_version"] == 1
            and resolved.get("object_type") == object_kind
            and resolved.get("object_id") == suffix
            and not any(resolved.get(field) for field in ("exception", "error", "errors")),
            "software_heritage: resolver response does not confirm the recorded object",
        )
        release = receipt.get("release")
        require(isinstance(release, dict), "software_heritage: selected release evidence required")
        require(nonempty(release.get("tag")), "software_heritage: selected tag required")
        require(hex_digest(release.get("commit"), 40), "software_heritage: selected release commit required")
        if identifier.startswith("swh:1:rev:"):
            require(
                identifier.endswith(":" + release["commit"]), "software_heritage: revision and release commit differ"
            )
    else:
        require(identifier.encode() in response, f"{key}: identifier absent from retained provider response")


def validate_packet(root: Path, packet: object) -> dict:
    """Validate metadata snapshots and recorded states without promoting release readiness."""
    require(isinstance(packet, dict), "Readiness manifest must be an object")
    require(type(packet.get("schema_version")) is int and packet["schema_version"] == 1, "Unsupported schema_version")
    require(packet.get("repository") == REPOSITORY, "Readiness repository identity mismatch")
    require(packet.get("parent_issue") == 398, "Readiness parent issue must be #398")
    require(hex_digest(packet.get("assessed_revision"), 40), "An immutable assessed_revision is required")
    version = tomllib.loads(read_local(root, "pyproject.toml").decode())["project"]["version"]
    require(packet.get("software_version") == version, "Readiness software_version differs from the current manifest")
    fingerprints = packet.get("input_sha256")
    require(isinstance(fingerprints, dict) and set(fingerprints) == INPUTS, "Required input fingerprint set changed")
    for path, digest in fingerprints.items():
        require(hex_digest(digest, 64), f"Input digest required: {path}")
        read_local(root, path, digest)
    entries = packet.get("destinations")
    require(
        isinstance(entries, list) and len(entries) == len(ISSUES), "Exactly three registry destinations are required"
    )
    keys = [entry.get("key") for entry in entries if isinstance(entry, dict)]
    require(all(isinstance(key, str) for key in keys) and set(keys) == set(ISSUES), "Missing or duplicate destination")
    pending = []
    for entry in entries:
        key, state = entry["key"], entry.get("state")
        require(entry.get("issue") == ISSUES[key], f"{key}: wrong child issue")
        require(
            isinstance(state, str) and state in {"pending", "submitted", "verified", "ineligible"},
            f"{key}: invalid state",
        )
        read_local(root, entry.get("dossier"))
        outstanding = entry.get("outstanding")
        require(
            isinstance(outstanding, list) and all(nonempty(item) for item in outstanding),
            f"{key}: invalid outstanding work",
        )
        terminal = state in {"verified", "ineligible"}
        require(bool(outstanding) != terminal, f"{key}: outstanding work conflicts with {state} state")
        if state == "pending":
            require(
                entry.get("identifier") is None and entry.get("evidence") is None,
                f"{key}: pending is not a provider outcome",
            )
        else:
            validate_evidence(root, packet, entry)
        if not terminal:
            pending.append(ISSUES[key])
    return {
        "schema_version": 1,
        "record_valid": True,
        "software_version": version,
        "assessed_revision": packet["assessed_revision"],
        "recorded_complete": not pending,
        "online_verification_performed": False,
        "pending_issues": sorted(pending),
        "destinations": [{key: entry[key] for key in ("key", "issue", "state", "outstanding")} for entry in entries],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (defaults to this checkout)")
    parser.add_argument("--manifest", default=MANIFEST, help="manifest path relative to the repository")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument(
        "--require-complete", action="store_true", help="fail if any registry has no recorded disposition"
    )
    args = parser.parse_args(argv)
    try:
        report = validate_packet(args.root, read_json(read_local(args.root, args.manifest)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"record_valid": False, "online_verification_performed": False, "error": str(exc)}
        code = 2
    else:
        code = 1 if args.require_complete and not report["recorded_complete"] else 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif code == 2:
        print(f"Invalid registry record: {report['error']}")
    else:
        print(f"Registry evidence record valid; unresolved issues: {report['pending_issues']}")
        print("Offline integrity checks only; external acceptance and release qualification are separate.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
