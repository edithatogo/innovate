"""Reject stale, misleading and incomplete source-derived learning catalogues."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/generate_learning_catalogue.py"
SPEC = importlib.util.spec_from_file_location("learning_catalogue", MODULE)
assert SPEC is not None and SPEC.loader is not None
catalogue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalogue)


class LearningCatalogueTests(unittest.TestCase):
    """Exercise independent synthetic inventories without importing Innovate."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.discovery = {
            "schema_version": "2.0",
            "metadata": {"runtime": "rust_native", "scope": "Synthetic source inventory"},
            "models": [
                {
                    "key": "logistic",
                    "family": "diffusion",
                    "stability": "experimental",
                    "supported_operations": ["fit_model", "predict_model"],
                    "supported_backends": ["rust"],
                    "simulation_engines": ["deterministic_curve"],
                    "simulation_semantics": "deterministic_curve_only",
                    "variants": [],
                    "supports_rates": False,
                    "supports_covariates": True,
                }
            ],
        }
        self.contract = {
            "schema_version": 1,
            "pages": {
                "user-guide/first.mdx": {"kind": "tutorial", "example": "examples/first.py"},
                "tutorials/again.mdx": {"kind": "tutorial", "example": "examples/first.py"},
                "old.md": {"kind": "historical"},
            },
            "examples": {
                "examples/first.py": {
                    "status": "runnable",
                    "language": "python",
                    "network": False,
                    "timeout_seconds": 60,
                    "expected_stdout": "checked\n",
                }
            },
        }
        self.write("scripts/generate_learning_catalogue.py", MODULE.read_text())
        self.write("examples/first.py", 'assert 2 + 2 == 4\nprint("checked")\n')
        self.write_page("user-guide/first.mdx", "First native result")
        self.write_page("tutorials/again.mdx", "Interpret the same result")
        self.write_page("old.md", "Old notebook interface")
        self.write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_page(self, relative: str, title: str) -> None:
        self.write(
            f"docs/astro-site/src/content/docs/{relative}",
            f"---\ntitle: {title}\ndescription: Interpret a bounded native example.\n---\n\nContent.\n",
        )

    def write_inputs(self) -> None:
        self.write("rust/crates/innovate-core/src/discovery.json", json.dumps(self.discovery))
        self.write("docs/contracts/content-quality.json", json.dumps(self.contract))

    def test_preserves_metadata_without_inventing_qualification_or_cartesian_support(self) -> None:
        result = catalogue.build_catalogue(self.root)
        self.assertEqual(result["models"][0]["advertised"], self.discovery["models"][0])
        self.assertIs(result["models"][0]["advertised"]["supports_rates"], False)
        self.assertNotIn("supports_parameter_events", result["models"][0]["advertised"])
        self.assertEqual(result["evidence_scope"]["native_execution"], "not_run_by_generator")
        self.assertEqual(result["evidence_scope"]["scientific_acceptance"], "not_established")
        self.assertNotIn("qualified_operations", result["models"][0])

    def test_counts_tutorials_and_distinct_examples_separately_and_excludes_history(self) -> None:
        result = catalogue.build_catalogue(self.root)
        self.assertEqual(result["counts"]["tutorials"], 2)
        self.assertEqual(result["counts"]["tutorial_example_sources"], 1)
        self.assertEqual(result["counts"]["registered_examples"], 1)
        self.assertEqual(len(result["examples"][0]["tutorial_ids"]), 2)
        self.assertNotIn("old", {item["id"] for item in result["tutorials"]})
        self.assertEqual(result["tutorials"][0]["route"], "/innovate/tutorials/again/")

    def test_added_current_tutorial_is_discovered_without_an_editorial_allowlist(self) -> None:
        self.contract["pages"]["tutorials/new.mdx"] = {"kind": "tutorial", "example": "examples/first.py"}
        self.write_page("tutorials/new.mdx", "A new task")
        self.write_inputs()
        self.assertEqual(catalogue.build_catalogue(self.root)["counts"]["tutorials"], 3)

    def test_duplicate_model_keys_fail(self) -> None:
        self.discovery["models"].append(copy.deepcopy(self.discovery["models"][0]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate model"):
            catalogue.build_catalogue(self.root)

    def test_new_model_requires_an_editorial_disposition(self) -> None:
        self.discovery["models"][0]["key"] = "unreviewed_model"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "editorial guidance"):
            catalogue.build_catalogue(self.root)

    def test_truthy_nonboolean_capability_is_rejected(self) -> None:
        self.discovery["models"][0]["supports_rates"] = "false"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "boolean"):
            catalogue.build_catalogue(self.root)

    def test_unknown_additive_metadata_is_preserved(self) -> None:
        self.discovery["models"][0]["new_boundary"] = {"maximum": 8}
        self.write_inputs()
        actual = catalogue.build_catalogue(self.root)["models"][0]["advertised"]["new_boundary"]
        self.assertEqual(actual, {"maximum": 8})

    def test_malformed_contract_structure_is_rejected(self) -> None:
        self.discovery["metadata"] = ["rust_native"]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "discovery contract"):
            catalogue.build_catalogue(self.root)
        self.discovery["metadata"] = {"runtime": "rust_native"}
        self.contract["pages"]["user-guide/first.mdx"]["examples"] = 3
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "list"):
            catalogue.build_catalogue(self.root)

    def test_missing_and_unregistered_tutorial_examples_fail(self) -> None:
        del self.contract["pages"]["user-guide/first.mdx"]["example"]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "one registered example"):
            catalogue.build_catalogue(self.root)
        self.contract["pages"]["user-guide/first.mdx"]["example"] = "examples/missing.py"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "unregistered example"):
            catalogue.build_catalogue(self.root)

    def test_missing_example_file_is_not_counted_as_available(self) -> None:
        (self.root / "examples/first.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            catalogue.build_catalogue(self.root)

    def test_duplicate_routes_and_unreviewed_slugs_fail(self) -> None:
        self.contract["pages"]["tutorials/again.md"] = {"kind": "reference"}
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate current documentation route"):
            catalogue.build_catalogue(self.root)
        del self.contract["pages"]["tutorials/again.md"]
        self.write_inputs()
        self.write(
            "docs/astro-site/src/content/docs/tutorials/again.mdx",
            "---\ntitle: Moved tutorial\ndescription: A custom URL.\nslug: elsewhere\n---\n",
        )
        with self.assertRaisesRegex(ValueError, "explicit slug"):
            catalogue.build_catalogue(self.root)

    def test_unsafe_paths_and_symlinks_fail(self) -> None:
        self.contract["pages"]["../outside.md"] = {"kind": "tutorial", "example": "examples/first.py"}
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            catalogue.build_catalogue(self.root)
        del self.contract["pages"]["../outside.md"]
        self.write_inputs()
        (self.root / "examples/first.py").unlink()
        (self.root / "examples/first.py").symlink_to(MODULE)
        with self.assertRaisesRegex(ValueError, "symlink"):
            catalogue.build_catalogue(self.root)

    def test_duplicate_json_fields_and_frontmatter_fail(self) -> None:
        self.write("docs/contracts/content-quality.json", '{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            catalogue.build_catalogue(self.root)
        self.write_inputs()
        self.write(
            "docs/astro-site/src/content/docs/tutorials/again.mdx",
            "---\ntitle: First\ntitle: Second\ndescription: Ambiguous title.\n---\n",
        )
        with self.assertRaisesRegex(ValueError, "duplicate frontmatter"):
            catalogue.build_catalogue(self.root)

    def test_render_is_deterministic_and_check_never_repairs_drift(self) -> None:
        catalogue.generate(self.root)
        output = self.root / catalogue.OUTPUT
        original = output.read_bytes()
        self.assertTrue(catalogue.generate(self.root, check=True))
        catalogue.generate(self.root)
        self.assertEqual(output.read_bytes(), original)
        self.write("examples/first.py", 'assert 3 + 3 == 6\nprint("checked")\n')
        self.assertFalse(catalogue.generate(self.root, check=True))
        self.assertEqual(output.read_bytes(), original)
        catalogue.generate(self.root)
        self.assertNotEqual(output.read_bytes(), original)

    def test_tampered_manifest_and_page_change_are_detected(self) -> None:
        catalogue.generate(self.root)
        output = self.root / catalogue.OUTPUT
        output.write_text(output.read_text().replace('"not_established"', '"accepted"'))
        self.assertFalse(catalogue.generate(self.root, check=True))
        catalogue.generate(self.root)
        self.write_page("tutorials/again.mdx", "Changed interpretation")
        self.assertFalse(catalogue.generate(self.root, check=True))

    def test_check_rejects_output_line_ending_drift_without_rewriting(self) -> None:
        catalogue.generate(self.root)
        output = self.root / catalogue.OUTPUT
        changed = output.read_bytes().replace(b"\n", b"\r\n")
        output.write_bytes(changed)
        self.assertFalse(catalogue.generate(self.root, check=True))
        self.assertEqual(output.read_bytes(), changed)


if __name__ == "__main__":
    unittest.main()
