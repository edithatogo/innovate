"""Reject ambiguous documentation without importing a scientific binding.

The synthetic runtime below exercises the runtime checker; it is not evidence
that a native extension, R package or Julia module has been executed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
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
                self.assertEqual(report["rust_workspace_scope"]["status"], "not_inspected")
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


class RustWorkspaceScopeTests(unittest.TestCase):
    """Reject scope drift without pretending to compile the fixture crates."""

    def setUp(self):
        self.source = load_checker("check_api_documentation")
        self.contract = self.source.default_contract()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.members = [
            "rust/crates/innovate-kairos",
            "rust/crates/innovate-core",
            "rust/crates/innovate-ffi",
            "rust/crates/innovate-python",
            "rust/crates/innovate-tabular",
        ]
        # Construct the fixture from the explicit reviewed exclusions. The
        # directory name is not necessarily the Cargo package name (e.g. qd).
        excluded = self.contract["rust"]["excluded_crates"]
        self.excluded = [Path(manifest).parent.as_posix() for manifest in excluded.values()]
        package_names = {Path(manifest).parent.as_posix(): name for name, manifest in excluded.items()}
        for directory in self.members + self.excluded:
            path = self.root / directory / "Cargo.toml"
            path.parent.mkdir(parents=True)
            name = package_names.get(directory, Path(directory).name)
            path.write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n', encoding="utf-8")
        self.write_workspace()
        package = self.root / "bindings/python/innovate/__init__.py"
        package.parent.mkdir(parents=True)
        package.write_text('"""Provide a documented module for structural scope testing."""\n', encoding="utf-8")

    def write_workspace(self):
        """Publish explicit fixture membership independently of the API policy."""
        (self.root / "Cargo.toml").write_text(
            f"[workspace]\nmembers = {json.dumps(self.members)}\nexclude = {json.dumps(self.excluded)}\n",
            encoding="utf-8",
        )

    def test_actual_workspace_and_fixture_membership_remain_unexecuted(self):
        for root in (ROOT, self.root):
            with self.subTest(root=root):
                report = self.source.audit(root, self.contract)
                scope = report["rust_workspace_scope"]
                self.assertEqual(scope["status"], "source_membership_matches")
                self.assertEqual(len(scope["workspace_crates"]), 5)
                self.assertIn("innovate-kairos", scope["workspace_crates"])
                self.assertNotIn("innovate-games", scope["workspace_crates"])
                self.assertEqual(scope["excluded_crates"], self.contract["rust"]["excluded_crates"])
                self.assertNotIn("innovate-games", scope["workspace_crates"])
                self.assertEqual(len(scope["excluded_crates"]), 17)
                self.assertEqual(scope["excluded_crates_qualification"], "not_qualified_by_workspace_commands")
                self.assertEqual(scope["compiler_and_doctests"], "not_executed")
                self.assertEqual(report["rust_status"], "not_executed")
                self.assertFalse(report["complete"])

    def test_stale_four_crate_contract_or_phantom_member_fails(self):
        for names in (
            [name for name in self.contract["rust"]["required_crates"] if name != "innovate-kairos"],
            [*self.contract["rust"]["required_crates"], "innovate-unlisted"],
        ):
            with self.subTest(names=names):
                changed = copy.deepcopy(self.contract)
                changed["rust"]["required_crates"] = names
                with self.assertRaisesRegex(ValueError, "required_crates differ"):
                    self.source.audit(self.root, changed)

    def test_package_name_comes_from_manifest_not_directory(self):
        path = self.root / "rust/crates/innovate-kairos/Cargo.toml"
        path.write_text('[package]\nname = "renamed-kairos"\nversion = "0.1.0"\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "required_crates differ"):
            self.source.audit(self.root, self.contract)

    def test_standalone_crate_cannot_disappear_or_be_promoted_by_policy(self):
        changed = copy.deepcopy(self.contract)
        changed["rust"]["excluded_crates"] = {}
        with self.assertRaisesRegex(ValueError, "excluded_crates differ"):
            self.source.audit(self.root, changed)
        changed["rust"]["required_crates"].append("innovate-games")
        with self.assertRaisesRegex(ValueError, "required_crates differ"):
            self.source.audit(self.root, changed)
        changed = copy.deepcopy(self.contract)
        changed["rust"]["excluded_crates_qualification"] = "passed"
        with self.assertRaisesRegex(ValueError, "separate compiler gates"):
            self.source.audit(self.root, changed)

    def test_changed_workspace_membership_and_unsupported_globs_fail(self):
        self.members.remove("rust/crates/innovate-kairos")
        self.write_workspace()
        with self.assertRaisesRegex(ValueError, "required_crates differ"):
            self.source.audit(self.root, self.contract)
        self.members = ["rust/crates/*"]
        self.write_workspace()
        with self.assertRaisesRegex(ValueError, "without globs or traversal"):
            self.source.audit(self.root, self.contract)

    def test_duplicate_scope_and_weakened_compiler_commands_fail(self):
        for field, value in (
            ("required_crates", [*self.contract["rust"]["required_crates"], "innovate-kairos"]),
            ("command", "cargo doc --workspace --no-deps --locked"),
            ("doctest_command", "cargo test --doc -p innovate-core --locked"),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed["rust"][field] = value
                with self.assertRaisesRegex(ValueError, "Rust documentation"):
                    self.source.audit(self.root, changed)


if __name__ == "__main__":
    unittest.main()
