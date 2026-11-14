"""Verify generated API coverage, reproducibility and bounded output ownership."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_api_reference.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("api_reference_generator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApiReferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generator = load_generator()
        self.package = self.root / "bindings/python/innovate"
        self.package.mkdir(parents=True)
        contract = self.root / self.generator.CONTRACT
        contract.parent.mkdir(parents=True)
        policy = json.loads((ROOT / self.generator.CONTRACT).read_bytes())
        policy["generated_reference_languages"] = ["python"]
        contract.write_text(json.dumps(policy), encoding="utf-8")
        self.output = self.root / self.generator.DEFAULT_OUTPUT
        self.write(self.complete_source())

    def write(self, text):
        (self.package / "__init__.py").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")

    def complete_source(self):
        return textwrap.dedent('''
            """Public measurement helpers with explicit source documentation."""
            __all__ = ["calculate", "Measurement"]

            def calculate(value: float, *, scale: float = 1.0) -> float:
                """Scale a measurement without changing its physical unit.

                Args:
                    value: The finite measurement supplied by the caller.
                    scale: A dimensionless multiplier, defaulting to one.
                Returns:
                    The scaled measurement in the original unit.
                Raises:
                    ValueError: If the input measurement is negative.
                Examples:
                    >>> calculate(2.0, scale=3.0)
                    6.0
                """
                if value < 0:
                    raise ValueError("negative measurement")
                return value * scale

            class Measurement:
                """Retain a measurement with its original numeric value.

                Attributes:
                    value: The finite measurement supplied by the caller.
                """
                value: float

                @property
                def unit(self) -> str:
                    """Read the fixed unit of this measurement.

                    Returns:
                        The unit name used for all stored values.
                    Raises:
                        None: Reading the fixed unit has no application-specific errors.
                    """
                    return "period"
        ''')

    def enable_polyglot(self):
        """Copy validated source fixtures without loading Julia, R or the native ABI."""
        checker = self.generator._polyglot_checker()
        inventory = checker.build_inventory(ROOT)
        for relative in inventory["sources"]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        contract = self.root / self.generator.CONTRACT
        policy = json.loads(contract.read_text())
        policy["generated_reference_languages"] = ["python", "julia", "r"]
        contract.write_text(json.dumps(policy), encoding="utf-8")
        self.record_polyglot(checker)
        return checker

    def record_polyglot(self, checker):
        """Refresh only this test's explicit inventory after an intentional mutation."""
        (self.root / checker.INVENTORY).write_text(checker.render_inventory(checker.build_inventory(self.root)))

    def test_language_policy_is_explicit_and_invalid_scope_fails(self):
        path = self.root / self.generator.CONTRACT
        policy = json.loads(path.read_text())
        for languages in (None, [], "python", ["python", "python"], ["python", "unknown"]):
            with self.subTest(languages=languages):
                policy["generated_reference_languages"] = languages
                path.write_text(json.dumps(policy))
                with self.assertRaisesRegex(ValueError, "generated_reference_languages"):
                    self.generator.build_artifacts(self.root)

    def test_enabled_languages_cannot_be_silently_skipped_when_sources_are_missing(self):
        path = self.root / self.generator.CONTRACT
        policy = json.loads(path.read_text())
        policy["generated_reference_languages"] = ["python", "julia", "r"]
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "source-file scope changed"):
            self.generator.build_artifacts(self.root)

    def test_polyglot_pages_cover_exact_inventory_with_separate_python_accounting(self):
        checker = self.enable_polyglot()
        expected = checker.build_inventory(self.root)
        artifacts = self.generator.build_artifacts(self.root)
        self.assertEqual(artifacts, self.generator.build_artifacts(self.root))
        prefix = self.generator.DEFAULT_OUTPUT.as_posix()
        manifest = json.loads(artifacts[f"{prefix}/manifest.json"])
        self.assertEqual(manifest["languages"]["python"]["symbol_count"], 4)
        self.assertEqual(manifest["languages"]["python"]["page_count"], 3)
        self.assertEqual(len(manifest["public_symbols"]), 4)
        self.assertEqual(manifest["polyglot"]["inputs"], expected["sources"])
        for language in ("julia", "r"):
            body = artifacts[f"{prefix}/{language}.md"]
            symbols = expected["symbols"][language]
            identities = {f"{item['source']}::{item['name']}" for item in symbols}
            self.assertEqual(set(manifest["languages"][language]["public_symbols"]), identities)
            self.assertEqual(set(manifest["pages"][f"{language}.md"]["symbols"]), identities)
            self.assertEqual(manifest["languages"][language]["symbol_count"], len(symbols))
            self.assertEqual(manifest["languages"][language]["page_count"], 1)
            self.assertEqual(manifest["languages"][language]["execution_status"], "not_run_by_generator")
            for symbol in symbols:
                self.assertIn(f"## `{symbol['name']}`", body)
                self.assertIn(symbol["source"], manifest["sources"])
            self.assertIn("### Arguments", body)
            self.assertIn("### Returns", body)
        self.assertIn("### Throws", artifacts[f"{prefix}/julia.md"])
        self.assertIn("### Fields", artifacts[f"{prefix}/julia.md"])
        self.assertIn("### Errors", artifacts[f"{prefix}/r.md"])
        self.assertIn("**`raw`**", artifacts[f"{prefix}/r.md"])

    def test_polyglot_source_drift_or_missing_recorded_inventory_fails(self):
        checker = self.enable_polyglot()
        source = self.root / checker.JULIA
        source.write_text(source.read_text() + "\n# Changed source documentation.\n")
        with self.assertRaisesRegex(ValueError, "stale or modified"):
            self.generator.build_artifacts(self.root)
        (self.root / checker.INVENTORY).unlink()
        with self.assertRaises(OSError):
            self.generator.build_artifacts(self.root)

    def test_removed_or_extra_exports_fail_against_validated_scope(self):
        checker = self.enable_polyglot()
        source = self.root / checker.JULIA
        original = source.read_text()
        self.assertIn("export invoke_registered_extension,", original)
        source.write_text(original.replace("export invoke_registered_extension,", "export", 1))
        with self.assertRaisesRegex(ValueError, "stale or modified"):
            self.generator.build_artifacts(self.root)
        source.write_text(original)
        namespace = self.root / "bindings/r/NAMESPACE"
        namespace.write_text(namespace.read_text() + "\nexport(undocumented_export)\n")
        with self.assertRaises(ValueError):
            self.generator.build_artifacts(self.root)

    def test_polyglot_generation_never_executes_language_source(self):
        checker = self.enable_polyglot()
        for relative, statement in (
            (checker.JULIA, 'error("must never execute")'),
            (checker.R_SOURCE, 'stop("must never execute")'),
        ):
            path = self.root / relative
            path.write_text(path.read_text() + "\n" + statement + "\n")
        self.record_polyglot(checker)
        self.assertIn((self.generator.DEFAULT_OUTPUT / "r.md").as_posix(), self.generator.build_artifacts(self.root))

    def test_language_routes_cannot_overwrite_python_symbol_pages(self):
        self.enable_polyglot()
        self.write(
            self.complete_source() + '\nclass Julia:\n    """A public class colliding with the language route."""\n'
        )
        with self.assertRaisesRegex(ValueError, "language reference collides"):
            self.generator.build_artifacts(self.root)

    def test_manually_modified_language_page_is_not_accepted_or_overwritten(self):
        self.enable_polyglot()
        self.generator.generate(self.root)
        page = self.output / "julia.md"
        page.write_text(page.read_text() + "\nUnreviewed rendered content.\n")
        before = page.read_bytes()
        self.assertTrue(self.generator.check_generated(self.root))
        with self.assertRaisesRegex(ValueError, "modified"):
            self.generator.generate(self.root)
        self.assertEqual(before, page.read_bytes())

    def test_deterministic_content_covers_signatures_and_class_members(self):
        first = self.generator.build_artifacts(self.root)
        self.assertEqual(first, self.generator.build_artifacts(self.root))
        prefix = self.generator.DEFAULT_OUTPUT.as_posix()
        reference = first[f"{prefix}/calculate.md"]
        self.assertIn("def calculate(value: float, *, scale: float=1.0) -> float", reference)
        self.assertIn("**`value`**", reference)
        self.assertIn("## Raises", reference)
        self.assertIn("calculate(2.0, scale=3.0)\n# Expected: 6.0", reference)
        self.assertIn("#L", reference)
        members = first[f"{prefix}/measurement.md"]
        self.assertIn("## `unit`", members)
        self.assertIn("@property", members)
        self.assertIn("value: float", members)
        manifest = json.loads(first[f"{prefix}/manifest.json"])
        rendered = {symbol for page in manifest["pages"].values() for symbol in page["symbols"]}
        self.assertEqual(rendered, set(manifest["public_symbols"]))
        self.assertEqual(len(manifest["sources"]), 1)
        self.assertEqual(len(manifest["generator_sha256"]), 64)
        self.assertIn("| --- | --- |\n|", first[f"{prefix}/index.md"])

    def test_never_imports_numerical_source(self):
        self.write(self.complete_source() + '\nraise RuntimeError("must never execute")\n')
        self.assertGreater(len(self.generator.build_artifacts(self.root)), 1)

    def test_missing_export_fails_before_creating_output(self):
        self.write(self.complete_source().replace('["calculate", "Measurement"]', '["missing"]'))
        with self.assertRaisesRegex(ValueError, "unresolved_export"):
            self.generator.generate(self.root)
        self.assertFalse(self.output.exists())

    def test_missing_docstring_or_parameter_contract_fails(self):
        self.write('''
            """A package with an undocumented public callable."""
            __all__ = ["calculate"]
            def calculate(value: float) -> float:
                return value
        ''')
        with self.assertRaisesRegex(ValueError, "missing_docstring"):
            self.generator.build_artifacts(self.root)
        self.write(self.complete_source().replace("value: The finite measurement supplied by the caller.", "value:"))
        with self.assertRaisesRegex(ValueError, "missing_parameter"):
            self.generator.build_artifacts(self.root)

    def test_exported_constant_without_renderable_documentation_is_rejected(self):
        source = self.complete_source().replace(
            '["calculate", "Measurement"]', '["calculate", "Measurement", "CONSTANT"]'
        )
        self.write(source + "\nCONSTANT = 4\n")
        with self.assertRaisesRegex(ValueError, "renderable definition"):
            self.generator.build_artifacts(self.root)

    def test_empty_public_scope_is_not_complete_reference(self):
        self.write('"""A documented package with no callable public surface."""\n__all__ = []')
        with self.assertRaisesRegex(ValueError, "empty"):
            self.generator.build_artifacts(self.root)

    def test_check_is_read_only_and_detects_stale_source(self):
        self.assertTrue(self.generator.check_generated(self.root))
        self.assertFalse(self.output.exists())
        self.generator.generate(self.root)
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.assertEqual(self.generator.check_generated(self.root), [])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        self.write(self.complete_source().replace("physical unit", "original unit"))
        self.assertTrue(any("stale" in error for error in self.generator.check_generated(self.root)))
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_cleanup_removes_only_manifest_owned_pages(self):
        self.generator.generate(self.root)
        unknown = self.output / "personal-notes.md"
        unknown.write_text("retain this unknown file", encoding="utf-8")
        revised = self.complete_source().split("class Measurement:")[0]
        revised = revised.replace('["calculate", "Measurement"]', '["calculate"]')
        self.write(revised)
        result = self.generator.generate(self.root)
        self.assertEqual(result["removed"], 1)
        self.assertFalse((self.output / "measurement.md").exists())
        self.assertEqual(unknown.read_text(), "retain this unknown file")
        self.assertEqual(self.generator.check_generated(self.root), [])

    def test_colliding_symbol_slugs_fail_instead_of_overwriting_a_page(self):
        self.write(
            self.complete_source()
            + '\nclass Calculate:\n    """A class with a colliding case-normalized page name."""\n'
        )
        with self.assertRaisesRegex(ValueError, "collide"):
            self.generator.build_artifacts(self.root)

    def test_modified_owned_page_is_preserved_instead_of_overwritten(self):
        self.generator.generate(self.root)
        page = self.output / "calculate.md"
        page.write_text(page.read_text() + "\nAn unreviewed manual edit.\n", encoding="utf-8")
        before = page.read_bytes()
        with self.assertRaisesRegex(ValueError, "modified"):
            self.generator.generate(self.root)
        self.assertEqual(page.read_bytes(), before)
        self.assertTrue(self.generator.check_generated(self.root))

    def test_unowned_conflicting_file_is_preserved(self):
        self.output.mkdir(parents=True)
        page = self.output / "calculate.md"
        page.write_text("existing authored content", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.generator.generate(self.root)
        self.assertEqual(page.read_text(), "existing authored content")
        self.assertFalse((self.output / "manifest.json").exists())

    def test_manifest_path_traversal_cannot_authorize_cleanup(self):
        self.generator.generate(self.root)
        protected = self.output.parent / "protected.md"
        protected.write_text("do not remove", encoding="utf-8")
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pages"]["../protected.md"] = {"sha256": "0" * 64}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.generator.generate(self.root)
        self.assertEqual(protected.read_text(), "do not remove")

    def test_symlinked_output_ancestor_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        parent = self.output.parent
        parent.parent.mkdir(parents=True)
        parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.generator.generate(self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_owned_page_is_preserved(self):
        self.generator.generate(self.root)
        outside = self.root / "outside.md"
        outside.write_text("keep outside content", encoding="utf-8")
        page = self.output / "calculate.md"
        page.unlink()
        page.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.generator.generate(self.root)
        self.assertEqual(outside.read_text(), "keep outside content")
        self.assertTrue(page.is_symlink())


if __name__ == "__main__":
    unittest.main()
