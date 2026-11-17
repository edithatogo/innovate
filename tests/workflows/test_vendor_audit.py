"""Verify that upstream advisory coverage fails closed on inventory drift."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vendor_audit", ROOT / "scripts/check_vendor_audit.py")
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class VendorAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vendor = Path(self.temp.name)
        self.crate = self.vendor / "example-1.2.3"
        self.crate.mkdir()
        (self.crate / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "1.2.3"\n')
        self.inventory = {
            "schema_version": 1,
            "packages": [{"name": "example", "version": "1.2.3", "crate_sha256": "a" * 64}],
        }
        self.write_inventory()
        (self.vendor / "Cargo.lock").write_text(AUDIT.audit_lock(self.vendor))

    def write_inventory(self):
        (self.vendor / "UPSTREAM.json").write_text(json.dumps(self.inventory))

    def test_valid_inventory_uses_the_registry_identity(self):
        AUDIT.check_inventory(self.vendor)
        lock = (self.vendor / "Cargo.lock").read_text()
        self.assertIn('source = "registry+https://github.com/rust-lang/crates.io-index"', lock)

    def test_missing_or_uninventoried_source_fails_closed(self):
        (self.vendor / "unlisted-1.0.0").mkdir()
        with self.assertRaisesRegex(ValueError, "directories and upstream inventory differ"):
            AUDIT.check_inventory(self.vendor)
        (self.vendor / "unlisted-1.0.0").rmdir()
        (self.crate / "Cargo.toml").unlink()
        with self.assertRaises(FileNotFoundError):
            AUDIT.check_inventory(self.vendor)

    def test_manifest_identity_checksum_and_audit_lock_must_match(self):
        (self.crate / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "2.0.0"\n')
        with self.assertRaisesRegex(ValueError, "manifest identity differs"):
            AUDIT.check_inventory(self.vendor)
        (self.crate / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "1.2.3"\n')
        self.inventory["packages"][0]["crate_sha256"] = "not-a-checksum"
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "invalid upstream checksum"):
            AUDIT.check_inventory(self.vendor)
        self.inventory["packages"][0]["crate_sha256"] = "b" * 64
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "audit lock differs"):
            AUDIT.check_inventory(self.vendor)

    def test_bundle_preserves_distinct_notices_and_detects_later_changes(self):
        (self.crate / "LICENSE").write_text("Shared licence and copyright\n")
        second = self.vendor / "second-1.0.0"
        second.mkdir()
        (second / "LICENSE").write_text("Shared licence and copyright\n")
        (second / "COPYING.extra").write_text("Additional notice\n")
        bundle = AUDIT.license_bundle(self.vendor)
        self.assertEqual(bundle.count(b"Shared licence and copyright"), 1)
        self.assertIn(b"example-1.2.3/LICENSE", bundle)
        self.assertIn(b"second-1.0.0/LICENSE", bundle)
        self.assertIn(b"Additional notice", bundle)
        bundle_path = self.vendor.parent / "THIRD-PARTY-LICENSES"
        bundle_path.write_bytes(bundle)
        AUDIT.check_license_bundle(self.vendor, bundle_path)
        (second / "COPYING.extra").write_text("Updated notice\n")
        with self.assertRaisesRegex(ValueError, "licence bundle differs"):
            AUDIT.check_license_bundle(self.vendor, bundle_path)

    def test_local_patch_hash_and_path_are_checked(self):
        source = self.crate / "source.rs"
        source.write_bytes(b"patched source\n")
        self.inventory["packages"][0]["local_patches"] = [
            {
                "path": "source.rs",
                "upstream_sha256": "b" * 64,
                "patched_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
        self.write_inventory()
        AUDIT.check_inventory(self.vendor)
        source.write_bytes(b"different source\n")
        with self.assertRaisesRegex(ValueError, "local patch differs"):
            AUDIT.check_inventory(self.vendor)
        self.inventory["packages"][0]["local_patches"][0]["path"] = "../escape.rs"
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "invalid local patch path"):
            AUDIT.check_inventory(self.vendor)


if __name__ == "__main__":
    unittest.main()
