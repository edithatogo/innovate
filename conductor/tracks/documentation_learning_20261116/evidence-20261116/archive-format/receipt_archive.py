"""Reversibly archive retained JSON receipts with explicit SHA-256 values.

This transforms copies only. It does not change a producer, verifier, scanner,
allowlist, repository index, or Git history. Archives are not verifier receipts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

FORMAT = "innovate-retained-json-sha256-v1"
FILES = (
    "artifact-after-baseline.json",
    "artifact.json",
    "hosted-178779bd.json",
    "qualification-summary.json",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")
SEMANTIC_ENCODING = (
    "python-json:sort_keys=true,ensure_ascii=false,separators=(',',':'),allow_nan=false;UTF-8;no-final-newline"
)


def digest(data: bytes) -> str:
    """Return an explicitly labelled SHA-256 digest of exact bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def pairs(items):
    """Reject duplicate object keys instead of discarding evidence."""
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def parse(raw: bytes):
    """Parse strict UTF-8 JSON without duplicate keys or non-finite values."""

    def invalid_constant(value):
        """Reject non-finite extensions to JSON."""
        raise ValueError("Non-finite JSON constant: " + value)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)


def semantic_bytes(value) -> bytes:
    """Serialize the documented canonical semantic comparison form."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def serialize(value, profile) -> bytes:
    """Reconstruct bytes using an explicitly recorded serialization profile."""
    separators = {"default": None, "compact": (",", ":"), "spaced": (", ", ": ")}[profile["separators"]]
    text = json.dumps(
        value,
        indent=profile["indent"],
        ensure_ascii=profile["ensure_ascii"],
        sort_keys=False,
        separators=separators,
        allow_nan=False,
    )
    return (text.replace("\n", profile["line_ending"]) + profile["final_newline"]).encode("utf-8")


def find_profile(value, original: bytes):
    """Find a lossless supported profile or reject the original format."""
    # Preserve object insertion order. Fail rather than approximate an unknown
    # original serialization; byte-identical reconstruction is required.
    for indent in (2, 4, None):
        for ascii_only in (True, False):
            for separators in ("default", "compact", "spaced"):
                for ending in ("\n", "\r\n"):
                    for final in (ending, ""):
                        profile = {
                            "indent": indent,
                            "ensure_ascii": ascii_only,
                            "separators": separators,
                            "line_ending": ending,
                            "final_newline": final,
                            "preserve_object_order": True,
                        }
                        if serialize(value, profile) == original:
                            return profile
    raise ValueError("Unsupported original JSON serialization; no lossy normalization performed")


def encode(original: bytes, repository_path: str):
    """Label hexadecimal digests while preserving every original byte."""
    value = parse(original)
    if not isinstance(value, dict):
        raise ValueError("These retained producer receipts must be JSON objects")
    profile = find_profile(value, original)
    paths = []

    def tag(item, path=()):
        """Record each changed JSON path without altering existing labels."""
        if isinstance(item, dict):
            return {key: tag(part, path + (key,)) for key, part in item.items()}
        if isinstance(item, list):
            return [tag(part, path + (index,)) for index, part in enumerate(item)]
        if isinstance(item, str) and HEX.fullmatch(item):
            paths.append(list(path))
            return "sha256:" + item
        return item

    record = tag(value)
    archive = {
        "archive_format": FORMAT,
        "original_record": {
            "repository_path": repository_path,
            "media_type": "application/json",
            "encoding": "utf-8",
            "byte_length": len(original),
            "serialized_sha256": digest(original),
            "semantic_sha256": digest(semantic_bytes(value)),
            "semantic_encoding": SEMANTIC_ENCODING,
            "serialization": profile,
        },
        "tagged_digest_paths": paths,
        "record": record,
    }
    raw = json.dumps(archive, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if decode(raw) != original:
        raise AssertionError("Archive did not reconstruct the exact original receipt bytes")
    return raw


def decode(raw: bytes) -> bytes:
    """Reconstruct a receipt and verify semantic and serialized identities."""
    archive = parse(raw)
    if (
        set(archive) != {"archive_format", "original_record", "tagged_digest_paths", "record"}
        or archive["archive_format"] != FORMAT
    ):
        raise ValueError("Unsupported retained receipt archive")
    original = archive["original_record"]
    if original["semantic_encoding"] != SEMANTIC_ENCODING or original["encoding"] != "utf-8":
        raise ValueError("Unknown semantic or text encoding")
    value = copy.deepcopy(archive["record"])
    seen = set()
    for path in archive["tagged_digest_paths"]:
        marker = json.dumps(path, ensure_ascii=False)
        if not path or marker in seen:
            raise ValueError("Empty or duplicate digest path")
        seen.add(marker)
        parent = value
        for part in path[:-1]:
            parent = parent[part]
        tagged = parent[path[-1]]
        if not isinstance(tagged, str) or not tagged.startswith("sha256:") or not HEX.fullmatch(tagged[7:]):
            raise ValueError("Tagged digest has changed or lacks its algorithm")
        parent[path[-1]] = tagged[7:]
    if digest(semantic_bytes(value)) != original["semantic_sha256"]:
        raise ValueError("Original semantic hash does not match")
    reconstructed = serialize(value, original["serialization"])
    if len(reconstructed) != original["byte_length"] or digest(reconstructed) != original["serialized_sha256"]:
        raise ValueError("Original serialized bytes do not match")
    return reconstructed


def main():
    """Encode copies, verify archives, or restore exact original receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("encode")
    write.add_argument("--source-dir", type=Path, required=True)
    write.add_argument("--output-dir", type=Path, required=True)
    write.add_argument("--repository-root", type=Path, required=True)
    write.add_argument("--files", nargs="+", default=FILES)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive-dir", type=Path, required=True)
    verify.add_argument("--original-dir", type=Path)
    verify.add_argument("--files", nargs="+", default=FILES)
    restore = commands.add_parser("decode")
    restore.add_argument("--input", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "encode":
        source = args.source_dir.resolve(strict=True)
        output = args.output_dir.resolve()
        if source == output:
            raise ValueError("Refusing to replace original receipts")
        output.mkdir(parents=True, exist_ok=True)
        for name in args.files:
            path = source / name
            original = path.read_bytes()
            relative = path.relative_to(args.repository_root.resolve(strict=True)).as_posix()
            with (output / name).open("xb") as stream:
                stream.write(encode(original, relative))
        print(json.dumps({"archived_copies": len(args.files), "originals_modified": False}))
    elif args.command == "decode":
        with args.output.open("xb") as stream:
            stream.write(decode(args.input.read_bytes()))
        print(json.dumps({"exact_original_bytes_reconstructed": True}))
    else:
        result = []
        for name in args.files:
            raw = (args.archive_dir / name).read_bytes()
            reconstructed = decode(raw)
            if args.original_dir is not None and reconstructed != (args.original_dir / name).read_bytes():
                raise ValueError("Supplied original differs: " + name)
            archive = parse(raw)
            result.append(
                {
                    "file": name,
                    "archive_sha256": digest(raw),
                    "original_serialized_sha256": archive["original_record"]["serialized_sha256"],
                    "original_semantic_sha256": archive["original_record"]["semantic_sha256"],
                    "tagged_digest_count": len(archive["tagged_digest_paths"]),
                    "exact_original_bytes_reconstructed": True,
                }
            )
        print(json.dumps({"archive_format": FORMAT, "verified": result}, indent=2))


if __name__ == "__main__":
    main()
