"""Adversarial runtime-inventory tests that do not import the numerical package."""

from __future__ import annotations

import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_native_documentation.py"
STUB = textwrap.dedent('''
    """Native fixture transport API for documentation tests."""
    def calculate(value: float, *, scale: float = 1.0) -> float:
        """Scale a supplied measurement in its original unit.

        Args:
            value: Finite measurement in the caller's original unit.
            scale: Dimensionless multiplier applied to the supplied measurement.
        Returns:
            The scaled measurement in the original unit.
        Raises:
            ValueError: If the supplied measurement is not finite.
            TypeError: If arguments cannot be converted to numeric values.
        """
        ...
''')
DOC = ast.get_docstring(ast.parse(STUB).body[1])


def load_checker():
    spec = importlib.util.spec_from_file_location("native_documentation_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeDocumentationTests(unittest.TestCase):
    def setUp(self):
        self.checker = load_checker()
        self.module = ModuleType("innovate._native", "Provide native transport operations for the scientific kernel.")

        def calculate(value, *, scale=1):
            return value * scale

        calculate.__doc__ = DOC
        self.module.calculate = calculate

    def report(self, stub=STUB):
        return self.checker.inspect_runtime(self.module, stub)

    def rules(self, report=None):
        return {gap["rule"] for gap in (report or self.report())["gaps"]}

    def test_matching_runtime_inventory_and_structured_docs_pass(self):
        report = self.report()
        self.assertEqual(report["gaps"], [])
        self.assertEqual(report["runtime_callables"], ["calculate"])
        self.assertEqual(report["stub_callables"], ["calculate"])

    def test_public_export_omitted_from_all_and_custom_dir_still_fails(self):
        self.module.extra = self.module.calculate
        self.module.__all__ = ["calculate"]
        self.module.__dir__ = lambda: ["calculate"]
        self.assertIn("undeclared_runtime_export", self.rules())
        self.assertIn("extra", self.report()["runtime_callables"])

    def test_underscore_export_in_all_cannot_escape_inventory(self):
        self.module._transport = self.module.calculate
        self.module.__all__ = ["calculate", "_transport"]
        self.assertIn("undeclared_runtime_export", self.rules())
        self.assertIn("_transport", self.report()["runtime_callables"])

    def test_explicit_underscore_export_with_a_stub_is_qualified(self):
        self.module._transport = self.module.calculate
        self.module.__all__ = ["calculate", "_transport"]
        stub = STUB + STUB.replace("calculate", "_transport")
        self.assertEqual(self.report(stub)["gaps"], [])

    def test_unexported_private_helpers_do_not_expand_public_contract(self):
        self.module._helper = lambda: None
        self.assertEqual(self.report()["gaps"], [])

    def test_pyo3_version_constant_in_all_is_recorded_separately(self):
        self.module.__version__ = "0.7.0a1"
        self.module.__all__ = ["calculate", "__version__"]
        report = self.report()
        self.assertEqual(report["gaps"], [])
        self.assertEqual(report["runtime_noncallable_exports"], ["__version__"])
        self.assertEqual(report["callable_count"], 1)

    def test_missing_declared_callable_fails(self):
        del self.module.calculate
        self.assertIn("missing_runtime_export", self.rules())
        self.assertIn("empty_callable_inventory", self.rules())

    def test_constant_cannot_replace_a_declared_function(self):
        self.module.calculate = "not callable"
        self.assertIn("noncallable_runtime_export", self.rules())

    def test_invalid_runtime_all_fails_closed(self):
        for exports in ("calculate", ["calculate", "calculate"], [42], ["not a name"]):
            with self.subTest(exports=exports):
                self.module.__all__ = exports
                self.assertIn("invalid_runtime_exports", self.rules())

    def test_missing_explicit_export_is_reported(self):
        self.module.__all__ = ["calculate", "missing_constant"]
        gaps = self.report()["gaps"]
        self.assertTrue(any(gap["symbol"] == "missing_constant" for gap in gaps))

    def test_dynamic_stub_export_lists_are_rejected(self):
        for extra in (
            "__all__ = choose_exports()\n",
            '__all__ = ["calculate"]\n__all__.append("other")\n',
            '__all__ = ["calculate"]\n__all__ += ["other"]\n',
            'if True:\n    __all__ = ["calculate"]\n',
        ):
            with self.subTest(extra=extra), self.assertRaises((ValueError, TypeError)):
                self.report(STUB + extra)

    def test_unresolved_stub_exports_and_empty_inventory_fail(self):
        for stub in ('"""Empty fixture transport module."""\n', STUB + '__all__ = ["unknown"]\n'):
            with self.subTest(stub=stub), self.assertRaises(ValueError):
                self.report(stub)

    def test_signature_or_placeholder_is_not_a_runtime_docstring(self):
        for doc in (
            None,
            "",
            "TODO document these arguments later.",
            "This documentation is a placeholder.",
            "calculate(value: float, scale: float = 1) -> float",
        ):
            with self.subTest(doc=doc):
                self.module.calculate.__doc__ = doc
                self.assertIn("non_substantive_runtime_docstring", self.rules())

    def test_runtime_parameters_return_and_error_sections_are_checked(self):
        self.module.calculate.__doc__ = "Scale the supplied measurement in its original unit."
        rules = self.rules()
        self.assertTrue(
            {
                "missing_runtime_parameter",
                "missing_runtime_returns",
                "missing_runtime_raises",
                "missing_runtime_exception",
            }
            <= rules
        )

    def test_unknown_documented_parameter_detects_signature_drift(self):
        self.module.calculate.__doc__ = DOC.replace("value:", "obsolete:")
        self.assertIn("unknown_runtime_parameter", self.rules())
        self.assertIn("missing_runtime_parameter", self.rules())

    def test_missing_runtime_exception_is_not_covered_by_another_error(self):
        self.module.calculate.__doc__ = DOC.replace("ValueError:", "RuntimeError:")
        gaps = self.report()["gaps"]
        self.assertTrue(
            any(gap["rule"] == "missing_runtime_exception" and gap["detail"] == "ValueError" for gap in gaps)
        )

    def test_explanatory_prose_after_google_errors_does_not_hide_valid_entries(self):
        self.module.calculate.__doc__ = DOC + "\n\nA synthetic result does not establish empirical validity."
        self.assertEqual(self.report()["gaps"], [])

    def test_unindented_postscript_cannot_supply_missing_error_entries(self):
        self.module.calculate.__doc__ = (
            DOC.replace("    TypeError: If arguments cannot be converted to numeric values.", "")
            + "\n\nTypeError: This unindented postscript is outside the structured error section."
        )
        self.assertTrue(
            any(
                gap["rule"] == "missing_runtime_exception" and gap["detail"] == "TypeError"
                for gap in self.report()["gaps"]
            )
        )

    def test_module_description_is_required(self):
        self.module.__doc__ = None
        self.assertIn("missing_module_docstring", self.rules())

    def test_builtin_extension_style_functions_are_not_skipped(self):
        self.module.calculate = len
        report = self.report()
        self.assertEqual(report["runtime_callables"], ["calculate"])
        self.assertNotIn("noncallable_runtime_export", self.rules(report))
        self.assertIn("missing_runtime_parameter", self.rules(report))

    def test_dynamic_public_names_in_dir_are_inventoried(self):
        self.module.__dir__ = lambda: ["calculate", "extra"]
        self.module.__getattr__ = lambda name: self.module.calculate if name == "extra" else None
        self.assertIn("undeclared_runtime_export", self.rules())


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.checker = load_checker()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "bindings/python/innovate"
        self.source.mkdir(parents=True)
        self.binding = self.source / "__init__.py"
        self.binding.write_text('"""Fixture Python binding for the runtime audit."""\n')
        (self.source / "_native.pyi").write_text(STUB)
        self.package = ModuleType("innovate")
        self.package.__file__ = str(self.binding)
        self.native = ModuleType("innovate._native", "Provide native transport operations for the scientific kernel.")
        self.extension = self.root / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])
        self.extension.write_bytes(b"fixture binary fingerprint; never dynamically loaded")
        self.native.__file__ = str(self.extension)
        loader = importlib.machinery.ExtensionFileLoader("innovate._native", str(self.extension))
        self.native.__spec__ = importlib.util.spec_from_loader("innovate._native", loader)

        def calculate(value, *, scale=1):
            return value * scale

        calculate.__doc__ = DOC
        self.native.calculate = calculate
        self.imports = {"innovate": self.package, "innovate._native": self.native}

    def audit(self):
        with patch.object(self.checker.importlib, "import_module", side_effect=self.imports.__getitem__):
            return self.checker.audit(self.root)

    def test_matched_sources_pass_but_do_not_attest_native_build_or_full_quality(self):
        report = self.audit()
        self.assertEqual(report["status"], "passed", report["gaps"])
        self.assertTrue(report["native_documentation_complete"])
        self.assertTrue(report["installed_source_matches"])
        self.assertFalse(report["native_binary_source_binding_verified"])
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["installed_extension"]["sha256"]), 64)

    def test_stale_installed_python_source_fails(self):
        installed = self.root / "installed.py"
        installed.write_text('"""Different installed Python binding source."""\n')
        self.package.__file__ = str(installed)
        report = self.audit()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["installed_source_matches"])
        self.assertIn("installed_source_mismatch", {gap["rule"] for gap in report["gaps"]})

    def test_python_module_cannot_impersonate_the_compiled_extension(self):
        self.native.__spec__ = importlib.util.spec_from_loader("innovate._native", loader=None)
        report = self.audit()
        self.assertEqual(report["status"], "failed")
        self.assertIn("compiled-extension loader", report["gaps"][-1]["detail"])

    def test_source_symlink_is_not_accepted_as_checkout_evidence(self):
        target = self.root / "target.py"
        target.write_bytes(self.binding.read_bytes())
        self.binding.unlink()
        self.binding.symlink_to(target)
        self.assertEqual(self.audit()["status"], "failed")

    def test_import_failure_overwrites_an_old_pass_receipt(self):
        output = self.root / "result.json"
        output.write_text('{"status":"passed"}')
        with (
            patch.object(self.checker.importlib, "import_module", side_effect=ImportError("native package absent")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = self.checker.main(["--root", str(self.root), "--output", str(output)])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_invalid_stub_fails_before_import_and_replaces_stale_evidence(self):
        (self.source / "_native.pyi").write_text("def broken(:\n")
        output = self.root / "result.json"
        output.write_text('{"status":"passed"}')
        with (
            patch.object(self.checker.importlib, "import_module") as imported,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = self.checker.main(["--root", str(self.root), "--output", str(output)])
        self.assertEqual(result, 1)
        imported.assert_not_called()
        self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_source_change_during_inspection_invalidates_result(self):
        original = self.checker.inspect_runtime

        def change_source(module, stub):
            self.binding.write_text('"""Changed while runtime inspection was in progress."""\n')
            return original(module, stub)

        with patch.object(self.checker, "inspect_runtime", side_effect=change_source):
            report = self.audit()
        self.assertEqual(report["status"], "failed")
        self.assertIn("source_changed_during_audit", {gap["rule"] for gap in report["gaps"]})


if __name__ == "__main__":
    unittest.main()
