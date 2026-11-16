"""Q04 must not confuse source references with release evidence."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("capability_claims", ROOT / "scripts/validate_capability_claims.py")
assert SPEC is not None and SPEC.loader is not None
claims = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(claims)


def test_unqualified_checked_in_release_claims_fail_closed():
    data = json.loads((ROOT / "conductor/capabilities.json").read_text())
    errors = claims.validate_capabilities(data, root=ROOT)
    assert any("native_receipt" in error and "source-bound" in error for error in errors)
    assert any("revision" in error for error in errors)
