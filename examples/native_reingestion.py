"""Verify both dataset encodings, complete-artifact replay and tamper rejection."""

import json
from copy import deepcopy

from innovate import NativeError, execute


def main() -> None:
    """Re-ingest a synthetic adoption dataset while preserving its checksum format."""
    dataset = {
        "kind": "adoption",
        "time": [0, 1],
        "adoption": [1, 2],
        "provenance": {
            "source": "synthetic survey α",
            "license": "CC0-1.0",
            "extraction_time": "2026-11-15T00:00:00Z",
        },
    }
    encodings = ["python_sorted_json_v1", "rust_sorted_json_v1"]
    rejected_count = 0
    for encoding in encodings:
        artifact = execute("ingest_dataset", dataset=dataset, encoding=encoding)["result"]
        assert artifact["checksum_encoding"] == encoding
        verified = execute("ingest_dataset", artifact=artifact)["result"]
        assert verified == artifact
        assert verified["dataset"]["adoption"] == [1, 2]
        assert verified["dataset"]["time"] == [0, 1]
        tampered = deepcopy(artifact)
        tampered["dataset"]["adoption"][1] = 3
        rejected = False
        try:
            execute("ingest_dataset", artifact=tampered)
        except NativeError as error:
            rejected = "checksum does not match" in str(error)
        assert rejected
        rejected_count += int(rejected)
    assert dataset["adoption"] == [1, 2]
    print(json.dumps({"encodings": encodings, "tampered_artifacts_rejected": rejected_count}, sort_keys=True))


if __name__ == "__main__":
    main()
