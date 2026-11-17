"""Behavioral tests for source-derived Python documentation qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_api_documentation.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("api_documentation_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocumentationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "bindings/python/innovate"
        self.package.mkdir(parents=True)
        self.checker = load_checker()

    def write(self, source, name="__init__.py"):
        path = self.package / name
        path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
        return path

    def report(self):
        return self.checker.audit(self.root, self.checker.default_contract())

    def rules(self, report=None):
        return {gap["rule"] for gap in (report or self.report())["gaps"]}

    def complete_source(self):
        return textwrap.dedent('''
            """A completely documented demonstration package."""
            __all__ = ["calculate"]
            def calculate(value: float, *, scale: float = 1) -> float:
                """Scale the supplied measurement without changing its unit.

                Args:
                    value: A finite input measurement in the caller's unit.
                    scale: A dimensionless multiplier for the measurement.
                Returns:
                    The scaled value in the original measurement unit.
                Raises:
                    ValueError: If the input measurement is negative.
                """
                if value < 0:
                    raise ValueError("negative value")
                return value * scale
        ''')

    def test_complete_python_is_not_whole_api_acceptance(self):
        self.write(self.complete_source())
        report = self.report()
        self.assertTrue(report["python_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["rust_status"], "not_executed")

    def test_never_imports_inventory_sources(self):
        self.write(self.complete_source() + '\nraise RuntimeError("must not import")\n')
        self.assertTrue(self.report()["python_complete"])

    def test_private_nested_and_stub_symbols_are_inventoried(self):
        self.write("""
            def public():
                def local():
                    pass
                pass
            def _helper():
                pass
        """)
        self.write("def transport(value: str) -> str: ...", "_native.pyi")
        report = self.report()
        exposure = {item["name"]: item["exposure"] for item in report["symbols"]}
        self.assertEqual(exposure["innovate.public.local"], "nested")
        self.assertEqual(exposure["innovate._helper"], "private")
        self.assertIn("innovate._native.transport", exposure)
        self.assertEqual(sum(gap["rule"] == "missing_docstring" for gap in report["gaps"]), 6)

    def test_parameter_mentions_in_summary_are_not_contracts(self):
        self.write('''
            """A demonstration of incomplete parameter documentation."""
            def calculate(value: float, *, scale: float = 1) -> float:
                """Multiply value and scale, then return the result.

                Returns:
                    The scaled result in the input unit.
                Raises:
                    None: This operation promises no library-specific errors.
                """
                return value * scale
        ''')
        missing = [g["detail"] for g in self.report()["gaps"] if g["rule"] == "missing_parameter"]
        self.assertEqual(missing, ["scale", "value"])

    def test_empty_parameter_and_return_sections_fail(self):
        self.write('''
            """A demonstration of empty documentation sections."""
            def calculate(value: float) -> float:
                """Calculate a scaled result from the measurement.

                Args:
                    value:
                Returns:
                Raises:
                """
                return value
        ''')
        self.assertTrue({"missing_parameter", "missing_returns", "missing_raises"} <= self.rules())

    def test_signature_drift_and_ghost_parameters_fail(self):
        self.write(
            self.complete_source()
            .replace("scale: float = 1", "multiplier: float = 1")
            .replace("value * scale", "value * multiplier")
        )
        report = self.report()
        self.assertTrue({"missing_parameter", "unknown_parameter"} <= self.rules(report))

    def test_direct_raise_requires_named_exception(self):
        self.write(self.complete_source().replace("ValueError: If the input", "TypeError: If the input"))
        self.assertIn("missing_exception", self.rules())

    def test_nested_raises_do_not_leak_to_outer_callable(self):
        self.write('''
            """A demonstration package with a local implementation helper."""
            def outer() -> int:
                """Return a fixed measurement without invoking the helper.

                Returns:
                    A fixed integer measurement for this example.
                Raises:
                    None: This operation promises no library-specific errors.
                """
                def helper():
                    """Raise the helper's deliberately isolated validation error."""
                    raise ValueError("local only")
                return 1
        ''')
        self.assertNotIn("missing_exception", self.rules())

    def test_numpy_style_parameters_returns_and_raises_pass(self):
        self.write('''
            """A documented package using the NumPy section convention."""
            def calculate(value: float) -> float:
                """Return the measurement in its original unit.

                Parameters
                ----------
                value : float
                    Input measurement in the caller's unit.
                Returns
                -------
                float
                    Unchanged measurement in the original unit.
                Raises
                ------
                ValueError
                    If the measurement is below zero.
                """
                if value < 0:
                    raise ValueError("negative")
                return value
        ''')
        self.assertTrue(self.report()["python_complete"])

    def test_constructor_and_property_are_public(self):
        self.write('''
            """A demonstration of public constructor and property inspection."""
            class Result:
                """Represent a result returned by the native scientific core."""
                def __init__(self, value):
                    self.value = value
                @property
                def measured(self) -> float:
                    return self.value
        ''')
        report = self.report()
        public_names = {s["name"] for s in report["symbols"] if s["exposure"] == "public"}
        self.assertIn("innovate.Result.__init__", public_names)
        self.assertIn("innovate.Result.measured", public_names)

    def test_class_fields_require_attribute_documentation(self):
        self.write('''
            """A demonstration of dataclass field documentation coverage."""
            class Result:
                """Represent a measurement result with explicit native metadata."""
                value: float
        ''')
        self.assertIn("missing_attribute", self.rules())

    def test_undefined_exports_are_not_silently_dropped(self):
        self.write('''
            """A demonstration of an unresolved public export."""
            __all__ = ["missing_name"]
        ''')
        self.assertIn("unresolved_export", self.rules())

    def test_unverified_diagnostic_details_are_fingerprinted_in_reports(self):
        export_names = ["diagnostic_fixture_alpha", "diagnostic_fixture_beta"]
        self.write(f'"""A demonstration package with invalid exports."""\n__all__ = {export_names!r}\n')
        report = self.report()
        gaps = [gap for gap in report["gaps"] if gap["rule"] == "unresolved_export"]
        self.assertEqual(len(gaps), 2)
        self.assertEqual(
            {gap["detail"] for gap in gaps},
            {"sha256:" + hashlib.sha256(name.encode()).hexdigest() for name in export_names},
        )
        self.assertEqual(len({gap["id"] for gap in gaps}), 2)
        self.assertFalse(report["inventory_valid"])

        output = self.root / "receipt.json"
        command = [sys.executable, str(SCRIPT), "--root", str(self.root)]
        for mode in ("strict", "audit"):
            for extra in ([], ["--output", str(output)]):
                result = subprocess.run(command + ["--mode", mode] + extra, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                published = result.stdout + result.stderr + (output.read_text() if extra else "")
                for name in export_names:
                    self.assertNotIn(name, published)

    def test_unresolved_export_fingerprint_handles_surrogates_and_empty_values(self):
        surrogate = "\ud800"
        self.write(f'"""A demonstration package with invalid exports."""\n__all__ = {[surrogate, ""]!r}\n')
        gaps = [gap for gap in self.report()["gaps"] if gap["rule"] == "unresolved_export"]
        self.assertEqual(len(gaps), 2)
        self.assertIn("", {gap["detail"] for gap in gaps})
        expected = hashlib.sha256(surrogate.encode("utf-8", errors="surrogatepass")).hexdigest()
        self.assertIn("sha256:" + expected, {gap["detail"] for gap in gaps})

    def test_unknown_parameter_fingerprint_preserves_ratchet_identity(self):
        marker = "diagnostic_fixture_parameter"
        self.write(self.complete_source().replace("scale: A dimensionless", f"{marker}: A dimensionless"))
        report = self.report()
        unknown = [gap for gap in report["gaps"] if gap["rule"] == "unknown_parameter"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["detail"], "sha256:" + hashlib.sha256(marker.encode()).hexdigest())
        self.assertNotIn(marker, json.dumps(report))
        baseline = self.checker.make_baseline(report)
        self.assertTrue(self.checker.evaluate_ratchet(report, baseline)["passed"])
        self.write(self.complete_source().replace("scale: A dimensionless", "another_parameter: A dimensionless"))
        self.assertFalse(self.checker.evaluate_ratchet(self.report(), baseline)["passed"])

    def test_legacy_baseline_values_do_not_escape_through_ratchet_receipts(self):
        marker = "diagnostic_fixture_parameter"
        self.write(self.complete_source().replace("scale: A dimensionless", f"{marker}: A dimensionless"))
        report = self.report()
        baseline = self.checker.make_baseline(report)
        for gap in baseline["gaps"]:
            if "::unknown_parameter:" in gap["id"]:
                gap["id"] = gap["id"].split("::unknown_parameter:")[0] + "::unknown_parameter:" + marker
        baseline["symbols"].append("untrusted_baseline_symbol_" + marker)
        baseline_path = self.root / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        receipt = self.root / "ratchet.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(self.root),
            "--mode",
            "ratchet",
            "--baseline",
            str(baseline_path),
        ]
        for extra in ([], ["--output", str(receipt)]):
            result = subprocess.run(command + extra, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            published = result.stdout + result.stderr + (receipt.read_text() if extra else "")
            self.assertNotIn(marker, published)
        check = self.checker.evaluate_ratchet(report, baseline)
        self.assertFalse(check["passed"])
        self.assertEqual(len(check["stale_gaps"]), 1)
        self.assertEqual(len(check["removed_symbols"]), 1)

    def test_imported_exports_require_explicit_source_qualification(self):
        self.write('''
            """A demonstration of an imported public export."""
            from unknown_package import imported
            __all__ = ["imported"]
        ''')
        self.assertIn("imported_export", self.rules())

    def test_imported_export_diagnostic_does_not_repeat_source_name(self):
        marker = "diagnostic_fixture_alias"
        self.write(
            '"""A demonstration of an imported public export."""\n'
            f"from module import item as {marker}\n"
            f"__all__ = [{marker!r}]"
        )
        report = self.report()
        imported = [gap for gap in report["gaps"] if gap["rule"] == "imported_export"]
        self.assertEqual(len(imported), 1)
        self.assertNotIn(marker, json.dumps(imported))
        self.assertTrue(imported[0]["detail"].startswith("sha256:"))

    def test_explicit_underscore_export_has_public_contract(self):
        self.write('''
            """A demonstration of explicit exports overriding name visibility."""
            __all__ = ["_transport"]
            def _transport(request: str) -> str:
                """Transport the supplied request through the native boundary."""
                return request
        ''')
        self.assertTrue({"missing_parameter", "missing_returns", "missing_raises"} <= self.rules())

    def test_conditional_and_async_nested_definitions_are_inventoried(self):
        self.write('''
            """A demonstration of definitions inside asynchronous control flow."""
            async def _outer():
                """Build local helpers inside an asynchronous context manager."""
                async with manager():
                    def hidden():
                        pass
        ''')
        names = {symbol["name"] for symbol in self.report()["symbols"]}
        self.assertIn("innovate._outer.hidden", names)

    def test_dynamic_exports_fail_closed(self):
        self.write('''
            """A demonstration of dynamically constructed public exports."""
            __all__ = list(globals())
        ''')
        self.assertIn("dynamic_exports", self.rules())

    def test_export_mutations_fail_closed(self):
        self.write('''
            """A demonstration of dynamically modified public exports."""
            __all__ = []
            __all__.append("hidden")
        ''')
        self.assertIn("dynamic_exports", self.rules())

    def test_missing_source_root_is_failure(self):
        self.package.rmdir()
        report = self.report()
        self.assertFalse(report["inventory_valid"])
        self.assertIn("missing_source_root", self.rules(report))

    def test_syntax_error_is_failure(self):
        self.write("def broken(")
        report = self.report()
        self.assertFalse(report["inventory_valid"])
        self.assertIn("source_parse_error", self.rules(report))

    def test_placeholder_docstring_is_not_documentation(self):
        self.write('''
            """TODO"""
            def _helper():
                """Documentation pending."""
        ''')
        self.assertIn("placeholder_docstring", self.rules())

    def test_ratchet_never_implies_completeness(self):
        self.write("def missing(value):\n    return value\n")
        report = self.report()
        baseline = self.checker.make_baseline(report)
        result = self.checker.evaluate_ratchet(report, baseline)
        self.assertTrue(result["passed"])
        self.assertFalse(report["python_complete"])
        self.assertFalse(report["complete"])

    def test_changed_deficient_symbol_invalidates_old_baseline(self):
        self.write("def missing(value):\n    return value\n")
        baseline = self.checker.make_baseline(self.report())
        self.write("def missing(value):\n    return value * 2\n")
        result = self.checker.evaluate_ratchet(self.report(), baseline)
        self.assertFalse(result["passed"])
        self.assertTrue(result["changed_gaps"])

    def test_line_shifts_do_not_invalidate_normalized_symbol_digest(self):
        self.write(self.complete_source() + "\n\ndef _helper():\n    pass\n")
        baseline = self.checker.make_baseline(self.report())
        path = self.package / "__init__.py"
        path.write_text("\n\n" + path.read_text(), encoding="utf-8")
        self.assertTrue(self.checker.evaluate_ratchet(self.report(), baseline)["passed"])

    def test_deleted_symbols_cannot_game_baseline(self):
        self.write(self.complete_source() + "\n\ndef _helper():\n    pass\n")
        baseline = self.checker.make_baseline(self.report())
        self.write(self.complete_source())
        result = self.checker.evaluate_ratchet(self.report(), baseline)
        self.assertFalse(result["passed"])
        removed = "bindings/python/innovate/__init__.py::innovate._helper"
        self.assertIn("sha256:" + hashlib.sha256(removed.encode()).hexdigest(), result["removed_symbols"])

    def test_duplicate_or_malformed_baseline_fails_closed(self):
        self.write("def missing(value):\n    return value\n")
        baseline = self.checker.make_baseline(self.report())
        baseline["gaps"].append(baseline["gaps"][0])
        self.assertFalse(self.checker.evaluate_ratchet(self.report(), baseline)["passed"])
        self.assertFalse(self.checker.evaluate_ratchet(self.report(), {"gaps": None})["passed"])

    def test_strict_cli_exit_status_is_not_audit_exit_status(self):
        self.write("def missing(value):\n    return value\n")
        report_path = self.root / "audit.json"
        base = [sys.executable, str(SCRIPT), "--root", str(self.root), "--output", str(report_path)]
        strict = subprocess.run(base + ["--mode", "strict"], capture_output=True, text=True)
        audit = subprocess.run(base + ["--mode", "audit"], capture_output=True, text=True)
        self.assertEqual(strict.returncode, 1, strict.stderr)
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertFalse(json.loads(report_path.read_text())["python_complete"])

    def test_malformed_contract_replaces_a_stale_pass_receipt(self):
        self.write(self.complete_source())
        contract = self.root / "contract.json"
        contract.write_text("null")
        output = self.root / "receipt.json"
        output.write_text(json.dumps({"python_complete": True, "command_passed": True}))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.root),
                "--contract",
                str(contract),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(output.read_text())
        self.assertFalse(report["python_complete"])
        self.assertFalse(report["command_passed"])

    def test_boolean_is_not_a_contract_version(self):
        self.write(self.complete_source())
        contract = self.checker.default_contract()
        contract["schema_version"] = True
        with self.assertRaises(ValueError):
            self.checker.audit(self.root, contract)

    def test_extended_placeholder_is_not_substantive_prose(self):
        self.write('''
            """TODO: Fill in this documentation later."""
        ''')
        self.assertIn("placeholder_docstring", self.rules())


    def test_repository_cargo_exclusions_match_explicit_rust_documentation_scope(self):
        """A workspace vendor exclusion is not evidence of a compiled Rust crate."""
        policy = self.checker.default_contract()["rust"]
        report = self.checker.rust_workspace_scope(ROOT, policy)
        self.assertEqual(report["status"], "source_membership_matches")
        self.assertEqual(set(report["workspace_crates"]), set(policy["required_crates"]))
        self.assertEqual(report["excluded_crates"], policy["excluded_crates"])
        self.assertEqual(
            report["excluded_crates"]["simba"],
            "rust/vendor/simba-0.9.1/Cargo.toml",
        )
        self.assertEqual(
            report["excluded_crates_qualification"],
            "not_qualified_by_workspace_commands",
        )
        # An omitted vendor manifest must cause a hard failure, not an
        # implicit qualification of its Rust API.
        incomplete = {
            **policy,
            "excluded_crates": {
                name: path for name, path in policy["excluded_crates"].items()
                if name != "simba"
            },
        }
        with self.assertRaisesRegex(ValueError, "excluded_crates differ"):
            self.checker.rust_workspace_scope(ROOT, incomplete)


if __name__ == "__main__":
    unittest.main()
