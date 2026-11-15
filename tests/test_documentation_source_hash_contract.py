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
                relative: "sha256:"
                + hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()
                for relative, text in self.sources.items()
            },
        }
        self.path = self.root / checker.POLYGLOT_INVENTORY
        self.path.parent.mkdir(parents=True)
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.inventory, indent=2) + "\n", encoding="utf-8")

    def native_receipt_fixture(self) -> Path:
        """Build complete synthetic receipts; these do not claim native execution."""
        for relative, text in {
            "bindings/julia/src/Innovate.jl": "# synthetic Julia source\r\n",
            "bindings/r/R/kernel_bridge.R": "# synthetic R source\n",
            "bindings/r/man/example.Rd": "Synthetic manual\n",
        }.items():
            source = self.root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(text.encode())
            digest = hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()
            self.inventory["sources"][relative] = "sha256:" + digest
        self.inventory.update(
            {
                "symbols": {language: [{"name": "example"}] for language in ("julia", "r")},
                "native_examples": {},
                "counts": {},
            }
        )
        native = self.root / "target/release/libinnovate_ffi.so"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_bytes(b"synthetic native identity")
        self.receipt_directory = self.root / "receipts"
        self.receipt_directory.mkdir(exist_ok=True)
        self.receipts = {}
        for language, source in (("julia", "bindings/julia/src/Innovate.jl"), ("r", "bindings/r/man/example.Rd")):
            code = "@assert 2 + 2 == 4\n" if language == "julia" else "stopifnot(2 + 2 == 4)\n"
            example = {
                "id": f"{language}:example:1",
                "source": source,
                "source_line": 1,
                "source_sha256": self.inventory["sources"][source].removeprefix("sha256:"),
                "code": code,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "assertion_count": 1,
            }
            self.inventory["native_examples"][language] = [example]
            self.inventory["counts"].update({f"{language}_example_blocks": 1, f"{language}_example_assertions": 1})
            receipt = {
                "schema_version": 1,
                "language": language,
                "status": "passed",
                "qualified": True,
                "runner": checker.LANGUAGE_RUNNERS[language],
                "runtime": "synthetic 1.0",
                "platform": "synthetic-test-platform",
                "package_version": "0.0-test",
                "source_sha256": dict(self.inventory["sources"]),
                "exported_names_checked": ["example"],
                "examples_executed": 1,
                "assertions_executed": 1,
                "native_library": {
                    "path": "/synthetic/libinnovate_ffi.so",
                    "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                },
                "examples": [
                    {key: example[key] for key in ("id", "source", "source_line", "code_sha256")}
                    | {"status": "passed", "assertions_executed": 1}
                ],
            }
            if language == "julia":
                receipt.update({"docstrings_checked": 1, "json_version": "0.0-test"})
            else:
                receipt.update(
                    {
                        "jsonlite_version": "0.0-test",
                        "manuals_checked": ["bindings/r/man/example.Rd"],
                        "installed_code_sha256": {
                            "/synthetic/innovate.rdb": "a" * 64,
                            "/synthetic/innovate.rdx": "b" * 64,
                        },
                    }
                )
            self.receipts[language] = receipt
        self.save_receipts()
        return self.receipt_directory

    def save_receipts(self) -> None:
        self.save()
        inventory_digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        for language, receipt in self.receipts.items():
            receipt["inventory_sha256"] = inventory_digest
            (self.receipt_directory / f"{language}-documentation.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )

    def test_complete_native_receipts_accept_mixed_source_encodings(self) -> None:
        directory = self.native_receipt_fixture()
        before = json.dumps(self.inventory, sort_keys=True)
        result = checker.verify_language_receipts(self.root, directory, ["python", "julia", "r"])
        for language in ("julia", "r"):
            with self.subTest(language=language):
                self.assertEqual(result[language]["status"], "passed", result[language])
                self.assertTrue(result[language]["qualified"])
                self.assertEqual(result[language]["receipt"]["source_sha256"], self.inventory["sources"])
        self.assertEqual(json.dumps(self.inventory, sort_keys=True), before)

    def test_complete_native_receipts_reject_changed_or_prefixed_example_hashes(self) -> None:
        directory = self.native_receipt_fixture()
        for language in ("julia", "r"):
            example = self.inventory["native_examples"][language][0]
            original = example["source_sha256"]
            for value in ("0" * 64, "sha256:" + original, "sha512:" + original, "SHA256:" + original, None):
                with self.subTest(language=language, value=value):
                    example["source_sha256"] = value
                    self.save_receipts()
                    result = checker.verify_language_receipts(self.root, directory, [language])[language]
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("Inventory example source hash is invalid", result["reason"])
            example["source_sha256"] = original

    def test_complete_native_receipts_reject_invalid_source_map_prefixes(self) -> None:
        directory = self.native_receipt_fixture()
        source = self.inventory["native_examples"]["julia"][0]["source"]
        digest = self.inventory["sources"][source].removeprefix("sha256:")
        for prefix in ("sha512:", "SHA256:", "sha256:sha256:"):
            with self.subTest(prefix=prefix):
                self.inventory["sources"][source] = prefix + digest
                for receipt in self.receipts.values():
                    receipt["source_sha256"] = dict(self.inventory["sources"])
                self.save_receipts()
                result = checker.verify_language_receipts(self.root, directory, ["julia", "r"])
                for item in result.values():
                    self.assertEqual(item["status"], "failed")
                    self.assertIn("Invalid polyglot source path or hash", item["reason"])

    def test_complete_native_receipts_preserve_exact_source_map_equality(self) -> None:
        directory = self.native_receipt_fixture()
        for language, receipt in self.receipts.items():
            with self.subTest(language=language):
                receipt["source_sha256"] = {
                    path: digest.removeprefix("sha256:") for path, digest in self.inventory["sources"].items()
                }
                self.save_receipts()
                result = checker.verify_language_receipts(self.root, directory, [language])[language]
                self.assertEqual(result["status"], "failed")
                self.assertIn("Receipt source hashes differ", result["reason"])

    def test_complete_native_receipts_still_rehash_source_bytes(self) -> None:
        directory = self.native_receipt_fixture()
        source = self.root / "bindings/julia/src/Innovate.jl"
        source.write_text("# source changed after receipt generation\n", encoding="utf-8")
        result = checker.verify_language_receipts(self.root, directory, ["julia", "r"])
        for item in result.values():
            self.assertEqual(item["status"], "failed")
            self.assertIn("Polyglot source changed", item["reason"])

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
