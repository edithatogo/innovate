"""Unit tests for R10 executable capability matrix validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_capability_claims import validate_capabilities


def test_checked_in_capabilities_matrix_is_valid():
    data = json.loads(Path("conductor/capabilities.json").read_text(encoding="utf-8"))
    assert validate_capabilities(data) == []


def test_absent_stale_or_missing_receipt_claims_are_rejected():
    data = json.loads(Path("conductor/capabilities.json").read_text(encoding="utf-8"))

    missing_native = copy.deepcopy(data)
    missing_native["capabilities"][0]["evidence"]["native_receipt"] = None
    assert validate_capabilities(missing_native)

    stale_rev = copy.deepcopy(data)
    assert validate_capabilities(stale_rev, expected_revision="0000000")

    missing_file = copy.deepcopy(data)
    missing_file["capabilities"][0]["evidence"]["scientific_receipt"] = "does/not/exist.rs"
    assert validate_capabilities(missing_file)
