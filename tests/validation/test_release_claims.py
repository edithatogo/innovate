"""Q04 executable release-claim truth tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_capability_claims import validate_capabilities


def test_release_claims_reject_stale_revision_and_missing_native_receipt():
    caps = json.loads(Path("conductor/capabilities.json").read_text(encoding="utf-8"))
    assert validate_capabilities(caps) == []

    stale = copy.deepcopy(caps)
    assert validate_capabilities(stale, expected_revision="stale_sha")

    unbacked = copy.deepcopy(caps)
    unbacked["capabilities"][0]["evidence"]["native_receipt"] = None
    assert validate_capabilities(unbacked)
