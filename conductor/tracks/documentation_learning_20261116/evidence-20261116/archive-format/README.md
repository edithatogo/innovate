# Explicit SHA-256 archives for retained evidence

This is an archived-evidence representation, not a change to any production verifier, receipt producer, secret scanner, allowlist, or CI gate. It was tested on a temporary one-commit repository containing all 49 files in the captured evidence directory. No repository source or history was changed.

## Result

Both Gitleaks 8.24.3 with its built-in rules and Gitleaks 8.30.1 with repository-standards@985503bc8eae58e7d6ef36a6e5427d0077e0193c/privacy/gitleaks.toml found 16 generic-api-key matches in the original snapshot. Each value was independently proved to be a file or retained-record SHA-256 digest. Both scanners returned exit code 0 and zero findings when only these four files used the archive envelope:

- artifact-after-baseline.json
- artifact.json
- hosted-178779bd.json
- qualification-summary.json

The other 45 files remained byte-identical. Scan commands, versions, checksum-verified download identities, all input fingerprints, finding paths and digest proofs are in verification.json. Captured scanner output is in scan-8.24.3.log and scan-8.30.1.log; their JSON findings reports are empty arrays. These are content/rule experiments, not a claim that a complete PR or any producer verifier has passed.

## Exact envelope schema

Each encoded file has exactly four top-level fields:

```json
{
  "archive_format": "innovate-retained-json-sha256-v1",
  "original_record": {
    "repository_path": "path/of/the/original/receipt.json",
    "media_type": "application/json",
    "encoding": "utf-8",
    "byte_length": 123,
    "serialized_sha256": "sha256:<original-file-digest>",
    "semantic_sha256": "sha256:<canonical-original-JSON-digest>",
    "semantic_encoding": "python-json:sort_keys=true,ensure_ascii=false,separators=(',',':'),allow_nan=false;UTF-8;no-final-newline",
    "serialization": {
      "indent": 2,
      "ensure_ascii": true,
      "separators": "default",
      "line_ending": "\n",
      "final_newline": "\n",
      "preserve_object_order": true
    }
  },
  "tagged_digest_paths": [["source_files", "example/path.py"]],
  "record": {"source_files": {"example/path.py": "sha256:<unchanged-digest>"}}
}
```

The example placeholders and byte count above illustrate the schema. Actual values and supported serialization profiles are recorded in each file.

Only complete lowercase 64-character hexadecimal JSON string values acquire the explicit sha256: tag. Every original field, path, digest and value remains present. tagged_digest_paths records exactly which values changed, including integer array indexes; pre-existing algorithm-tagged strings are not changed or accidentally stripped. Object insertion order is preserved. No base64, opaque compression, scanner suppression or excluded path is used.

The semantic hash is the defined Python JSON serialization, not an RFC 8785/JCS claim. The separate original serialization profile preserves the exact original UTF-8 bytes, including final newline and line endings. Encoding fails if the original bytes cannot be reconstructed by a supported profile. The normalizer verifies both semantic equality and byte-for-byte reconstruction before writing a copy.

## Reuse after final records are ready

Use a fresh output directory. The experiment in `verification.json` describes
the captured snapshot; regenerate archives after summary or artifact receipts
change. The original aggregate from that snapshot is now retained as
`../historical-summary-main69.json`; the current `../qualification-summary.json`
is a separate index of current and historical evidence.

```sh
python conductor/tracks/documentation_learning_20261116/evidence-20261116/archive-format/receipt_archive.py encode --source-dir conductor/tracks/documentation_learning_20261116/evidence-20261116 --output-dir /tmp/final-evidence-tagged --repository-root .
python conductor/tracks/documentation_learning_20261116/evidence-20261116/archive-format/receipt_archive.py verify --archive-dir /tmp/final-evidence-tagged --original-dir conductor/tracks/documentation_learning_20261116/evidence-20261116
```

The script refuses to overwrite existing output files or use the source directory as its output directory. No original receipt is modified by these commands. After choosing to copy the archives into the successor track, label them as retained archives and link this decoding contract. Re-run the unchanged scanners against the actual complete candidate commit; this snapshot experiment is not a substitute for that scan.

## Decode and verify

```sh
python conductor/tracks/documentation_learning_20261116/evidence-20261116/archive-format/receipt_archive.py decode --input /tmp/final-evidence-tagged/artifact.json --output /tmp/decoded-original-artifact.json
```

The decoder removes tags only at listed paths, verifies the canonical original-data hash, applies the recorded original serialization, and checks its length and exact byte hash. All four prepared archives were decoded and compared directly with their original source bytes.

The envelope itself must not be passed to the production verifier as if it were that verifier's receipt schema. The original schema is the decoded record. Cross-record digests retained inside record still identify the original decoded record bytes; archive-file hashes and original-record hashes are different identities and must be reported separately. Original CI run and artifact IDs remain provenance fields inside the unchanged decoded records.

The retained CLI adds function documentation, repository formatting, and a `--files` selection for later receipts. The original experiment records its own script hash. Each newly encoded copy must still pass exact byte reconstruction. This archive format never changes producer schemas or source verification.

## Integrated receipt verification

The fresh main582 receipt snapshot contains 62 files. Both unchanged scanners
identified 32 additional generic API-key matches in four files under
`../integrated-5821d8a/`: `final-source/artifact.json` (7),
`final-source/community.json` (4), `final-source/community.log` (4), and
`rust/innovate-rust-doc-token-evidence-20261116.json` (17).
The `.log` file is JSON stdout, byte-identical to the community JSON receipt.

[Source proofs](integrated-source-match-proof.json) verify the 15 source-map
matches against actual current file bytes. [Token proofs](integrated-token-match-proof.json)
reproduce all 17 token digests from the retained Rust tokenization helper and
check each recorded final source hash. The four matched receipts use the same
lossless archive envelope; the original copy manifest continues to identify decoded originals.
The root evidence index separately fingerprints the stored archive files.
The full staged candidate scan is recorded separately from this receipt snapshot.
