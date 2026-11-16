"""Reject ambiguous documentation without importing a scientific binding.

The synthetic runtime below exercises the runtime checker; it is not evidence
that a native extension, R package or Julia module has been executed.
"""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_checker(name):
    """Load only the dependency-free documentation tooling."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOOGLE_DOC = """Scale a supplied measurement in its original unit.

Args:
    value: Finite measurement in the caller's original unit.
    scale: Dimensionless multiplier applied to the supplied measurement.
Returns:
    The scaled measurement in the original unit.
Raises:
    ValueError: If the supplied measurement is not finite.
"""

NUMPY_DOC = """Scale a supplied measurement in its original unit.

Parameters
----------
value, scale : float
    Finite operands in the caller's declared units.
Returns
-------
float
    The scaled measurement in the original unit.
Raises
------
ValueError
    If the supplied measurement is not finite.
"""


class DocumentationConventionTests(unittest.TestCase):
    """Use conflicting descriptions that the old map-based parser discarded."""

    def setUp(self):
        self.source = load_checker("check_api_documentation")
        self.runtime = load_checker("check_native_documentation")
        self.polyglot = load_checker("check_polyglot_api_documentation")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "bindings/python/innovate"
        self.package.mkdir(parents=True)

    def fixture(self, doc):
        """Build source with a real signature and a caller-supplied docstring."""
        return (
            '"""A documentation contract fixture with no native dependency."""\n'
            "def measure(value: float, scale: float = 1.0) -> float:\n"
            + textwrap.indent('"""' + doc + '"""\n', "    ")
            + "    return value * scale\n"
        )

    def source_report(self, doc):
        (self.package / "__init__.py").write_text(self.fixture(doc), encoding="utf-8")
        return self.source.audit(self.root, self.source.default_contract())

    def runtime_report(self, doc):
        module = ModuleType("innovate._native", "Provide a synthetic documented measurement transport.")

        def measure(value, scale=1.0):
            return value * scale

        measure.__doc__ = doc
        module.measure = measure
        return self.runtime.inspect_runtime(module, self.fixture(GOOGLE_DOC))

    def test_supported_google_and_numpy_contracts_remain_valid(self):
        for doc in (GOOGLE_DOC, NUMPY_DOC):
            with self.subTest(style=doc.splitlines()[2]):
                report = self.source_report(doc)
                self.assertEqual(report["gaps"], [])
                self.assertTrue(report["python_complete"])
                self.assertFalse(report["complete"])
                self.assertEqual(self.runtime_report(doc)["gaps"], [])

    def test_aliased_sections_cannot_overwrite_an_earlier_contract(self):
        duplicate = GOOGLE_DOC.replace(
            "Args:\n", "Arguments:\n    value: Value measured in incompatible seconds.\nArgs:\n"
        )
        report = self.source_report(duplicate)
        self.assertFalse(report["python_complete"])
        self.assertIn(
            ("duplicate_docstring_section", "parameters"),
            {(gap["rule"], gap["detail"]) for gap in report["gaps"]},
        )

    def test_mixed_styles_cannot_hide_a_duplicate_return_contract(self):
        duplicate = GOOGLE_DOC.replace(
            "Returns:\n", "Returns\n-------\nfloat\n    A rate in events per second.\nReturns:\n"
        )
        self.assertIn("duplicate_docstring_section", {gap["rule"] for gap in self.source_report(duplicate)["gaps"]})

    def test_repeated_google_parameter_error_and_attribute_entries_fail(self):
        for section, name, original in (
            ("parameters", "value", "    value: Finite measurement in the caller's original unit.\n"),
            ("raises", "ValueError", "    ValueError: If the supplied measurement is not finite.\n"),
        ):
            with self.subTest(section=section):
                duplicate = GOOGLE_DOC.replace(original, original + original)
                self.assertIn(
                    ("duplicate_docstring_entry", f"{section}:{name}"),
                    {(gap["rule"], gap["detail"]) for gap in self.source_report(duplicate)["gaps"]},
                )
        doc = (
            GOOGLE_DOC
            + "\nAttributes:\n    unit: Unit of the original measurement.\n    unit: Unit of the final measurement.\n"
        )
        self.assertIn(("duplicate_docstring_entry", "attributes:unit"), self.source._docstring_issues(doc))

    def test_grouped_numpy_parameters_cannot_repeat_a_name(self):
        duplicate = NUMPY_DOC.replace(
            "Returns\n", "value : float\n    A conflicting description measured in seconds.\nReturns\n"
        )
        self.assertIn(
            ("duplicate_docstring_entry", "parameters:value"),
            {(gap["rule"], gap["detail"]) for gap in self.source_report(duplicate)["gaps"]},
        )

    def test_runtime_docs_use_the_same_ambiguity_gate(self):
        duplicate = GOOGLE_DOC + "\nArguments:\n    value: A competing value measured in seconds.\n"
        self.assertIn(
            ("duplicate_runtime_docstring_section", "parameters"),
            {(gap["rule"], gap["detail"]) for gap in self.runtime_report(duplicate)["gaps"]},
        )
        duplicate = GOOGLE_DOC.replace("    scale:", "    value: A competing value measured in seconds.\n    scale:")
        self.assertIn(
            ("duplicate_runtime_docstring_entry", "parameters:value"),
            {(gap["rule"], gap["detail"]) for gap in self.runtime_report(duplicate)["gaps"]},
        )

    def test_each_language_policy_fails_closed_when_changed(self):
        contract = self.source.default_contract()
        self.assertEqual(set(contract["language_conventions"]), {"python", "rust", "julia", "r"})
        for language in contract["language_conventions"]:
            with self.subTest(language=language):
                changed = copy.deepcopy(contract)
                changed["language_conventions"][language]["style"] = "unreviewed prose"
                with self.assertRaisesRegex(ValueError, "language conventions"):
                    self.source.audit(self.root, changed)
                changed = copy.deepcopy(contract)
                del changed["language_conventions"][language]
                with self.assertRaisesRegex(ValueError, "language conventions"):
                    self.source.audit(self.root, changed)
                changed = copy.deepcopy(contract)
                policy = changed["language_conventions"][language]
                if "structurally_enforced" in policy:
                    rule = next(iter(policy["structurally_enforced"]))
                    policy["structurally_enforced"][rule] = 1
                else:
                    policy["qualified_by_python_audit"] = 0
                with self.assertRaisesRegex(ValueError, "language conventions"):
                    self.source.audit(self.root, changed)

    def test_polyglot_policy_records_checks_separately_from_execution(self):
        contract = self.source.default_contract()
        for language, convention in self.polyglot.LANGUAGE_CONVENTIONS.items():
            self.assertEqual(contract["language_conventions"][language], convention)
            self.assertTrue(all(convention["structurally_enforced"].values()))
            changed = copy.deepcopy(contract)
            rule = next(iter(convention["structurally_enforced"]))
            changed["language_conventions"][language]["structurally_enforced"][rule] = False
            with self.assertRaisesRegex(ValueError, "language conventions"):
                self.source.audit(self.root, changed)


if __name__ == "__main__":
    unittest.main()
