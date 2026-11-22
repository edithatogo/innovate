"""Verify inventory selection from publication bytes without building native code."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("extract_wheel_sbom", ROOT / "scripts/extract_wheel_sbom.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WheelSbomTests(unittest.TestCase):
    def test_copies_only_the_embedded_inventory_without_changing_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "candidate.whl"
            payload = {
                "bomFormat": "CycloneDX",
                "components": [{"name": "serde", "version": "1.0.0", "purl": "pkg:cargo/serde@1.0.0"}],
            }
            with ZipFile(wheel, "w") as archive:
                archive.writestr("innovate.dist-info/sboms/rust.cyclonedx.json", json.dumps(payload))
                archive.writestr("unrelated.cyclonedx.json", "{}")
            before = wheel.read_bytes()
            MODULE.extract(wheel, root / "sbom.json")
            self.assertEqual(json.loads((root / "sbom.json").read_text()), payload)
            self.assertEqual(wheel.read_bytes(), before)

    def test_rejects_missing_ambiguous_or_empty_inventory(self):
        for count, components in ((0, []), (2, [{"purl": "pkg:cargo/a@1"}]), (1, [])):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                wheel = root / "candidate.whl"
                with ZipFile(wheel, "w") as archive:
                    for index in range(count):
                        archive.writestr(
                            f"innovate.dist-info/sboms/{index}.cyclonedx.json",
                            json.dumps({"bomFormat": "CycloneDX", "components": components}),
                        )
                with self.assertRaises(ValueError):
                    MODULE.extract(wheel, root / "sbom.json")
                self.assertFalse((root / "sbom.json").exists())

    def test_rejects_non_rust_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "candidate.whl"
            with ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "innovate.dist-info/sboms/runtime.cyclonedx.json",
                    json.dumps({"bomFormat": "CycloneDX", "components": [{"purl": "pkg:pypi/requests@2.0"}]}),
                )
            with self.assertRaises(ValueError):
                MODULE.extract(wheel, root / "sbom.json")


if __name__ == "__main__":
    unittest.main()
