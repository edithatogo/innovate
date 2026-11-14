"""Source hash encodings must agree with native runners without weakening rehashing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "documentation_hash_checker", ROOT / "scripts/check_documentation_content.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DocumentationSourceHashTests(unittest.TestCase):
    """Exercise the verifier against fresh, bounded synthetic source inventories."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sources = {
            str(checker.POLYGLOT_CHECKER): "def check_inventory(root):\n    return []\n",
            checker.LANGUAGE_RUNNERS["julia"]: "# synthetic Julia runner\r\n",
            checker.LANGUAGE_RUNNERS["r"]: "# synthetic R runner\r",
        }
        for relative, text in self.sources.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
        self.inventory = {
            "schema_version": 1,
            "sources": {
                relative: "sha256:" + hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()
                for relative, text in self.sources.items()
            },
        }
        self.path = self.root / checker.POLYGLOT_INVENTORY
        self.path.parent.mkdir(parents=True)
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.inventory, indent=2) + "\n", encoding="utf-8")

    def test_prefixed_native_source_hashes_pass_without_rewriting_inventory(self) -> None:
        inventory, digest = checker._verified_native_inventory(self.root)
        self.assertEqual(inventory, self.inventory)
        self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertTrue(all(value.startswith("sha256:") for value in inventory["sources"].values()))

    def test_existing_bare_sha256_encoding_remains_compatible(self) -> None:
        self.inventory["sources"] = {
            path: value.removeprefix("sha256:") for path, value in self.inventory["sources"].items()
        }
        self.save()
        inventory, _ = checker._verified_native_inventory(self.root)
        self.assertEqual(inventory, self.inventory)

    def test_source_rehashing_remains_mandatory(self) -> None:
        source = self.root / checker.LANGUAGE_RUNNERS["julia"]
        source.write_text("# changed source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Polyglot source changed"):
            checker._verified_native_inventory(self.root)

    def test_well_formed_but_wrong_digest_is_rejected(self) -> None:
        self.inventory["sources"][checker.LANGUAGE_RUNNERS["r"]] = "sha256:" + "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "Polyglot source changed"):
            checker._verified_native_inventory(self.root)

    def test_unknown_algorithms_and_malformed_hashes_are_rejected(self) -> None:
        digest = "a" * 64
        for value in (
            "sha512:" + digest,
            "SHA256:" + digest,
            "sha256:sha256:" + digest,
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            "sha256:" + "g" * 64,
            " sha256:" + digest,
            "sha256:" + digest + "\n",
            "sha256:",
            None,
            1,
            True,
            [],
            {},
        ):
            with self.subTest(value=value):
                self.inventory["sources"][checker.LANGUAGE_RUNNERS["r"]] = value
                self.save()
                with self.assertRaisesRegex(ValueError, "Invalid polyglot source path or hash"):
                    checker._verified_native_inventory(self.root)

    def test_relative_path_validation_remains_mandatory(self) -> None:
        self.inventory["sources"]["../outside.py"] = "sha256:" + "a" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "Invalid polyglot source path or hash"):
            checker._verified_native_inventory(self.root)

    def test_missing_source_is_not_qualified(self) -> None:
        (self.root / checker.LANGUAGE_RUNNERS["r"]).unlink()
        with self.assertRaisesRegex(ValueError, "Polyglot source is absent or external"):
            checker._verified_native_inventory(self.root)

    def test_checker_and_runner_inventory_remains_complete(self) -> None:
        del self.inventory["sources"][checker.LANGUAGE_RUNNERS["r"]]
        self.save()
        with self.assertRaisesRegex(ValueError, "Polyglot inventory omits a checker or native runner"):
            checker._verified_native_inventory(self.root)

    def test_native_binary_hash_format_is_not_broadened(self) -> None:
        self.assertTrue(checker._valid_hash("a" * 64))
        self.assertFalse(checker._valid_hash("sha256:" + "a" * 64))


if __name__ == "__main__":
    unittest.main()
