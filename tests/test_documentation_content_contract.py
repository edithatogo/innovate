"""Adversarial tests for content inventory, source binding and execution evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import venv
from contextlib import redirect_stdout
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/check_documentation_content.py"
SPEC = importlib.util.spec_from_file_location("content_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ContentContractTests(unittest.TestCase):
    """Check failures that a count-only documentation gate would miss."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.site = self.root / "docs/astro-site/src/content/docs"
        self.site.mkdir(parents=True)
        (self.root / "examples").mkdir()
        self.example = self.root / "examples/forecast.py"
        self.example.write_text('value = 2 + 2\nassert value == 4\nprint("checked")\n')
        self.page = self.site / "user-guide/quickstart.mdx"
        self.page.parent.mkdir()
        self.page.write_text(self.page_text())
        self.contract = {
            "schema_version": 1,
            "pages": {
                "user-guide/quickstart.mdx": {
                    "kind": "tutorial",
                    "example": "examples/forecast.py",
                }
            },
            "examples": {
                "examples/forecast.py": {
                    "status": "runnable",
                    "language": "python",
                    "network": False,
                    "timeout_seconds": 5,
                    "expected_stdout": "checked\n",
                }
            },
        }
        api_policy = self.root / "docs/contracts/api-documentation.json"
        api_policy.parent.mkdir(parents=True)
        api_policy.write_text(json.dumps({"generated_reference_languages": ["python"]}))

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def page_text() -> str:
        return """---
title: A reproducible forecast
description: Run and check a synthetic forecast with an explicit numerical oracle.
---
import { Code } from '@astrojs/starlight/components';
import forecastSource from '../../../../../../examples/forecast.py?raw';

## Learning outcomes
Run the complete example and explain the synthetic result.
## Prerequisites
Use the pinned Python runtime with the local package installed.
## Steps
Save the example and run the Python file in a clean directory.
<Code code={forecastSource} lang="python" />
## Expected results
The program prints checked after the independent equality assertion passes.
## Interpretation
The result establishes this deterministic calculation for the given inputs.
## Limitations
This synthetic calculation does not establish external empirical validity.
"""

    def report(self) -> dict:
        return checker.build_report(self.root, self.contract)

    def rules(self) -> set[str]:
        return {item["rule"] for item in self.report()["violations"]}

    def register_example(self, name: str, value: int) -> Path:
        """Create a distinct declared source for multiple-binding regression cases."""
        path = self.root / "examples" / name
        path.write_text(f'value = {value} + 1\nassert value == {value + 1}\nprint("checked")\n')
        self.contract["examples"][path.relative_to(self.root).as_posix()] = dict(
            self.contract["examples"]["examples/forecast.py"]
        )
        return path

    def test_valid_source_bound_tutorial_is_structurally_complete(self) -> None:
        report = self.report()
        self.assertEqual([], report["violations"])
        self.assertTrue(report["current_structure_complete"])
        self.assertTrue(report["full_scope_structure_complete"])
        self.assertFalse(report["current_content_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual("not_run", report["execution_status"])

    def test_new_page_cannot_hide_behind_legacy_notice(self) -> None:
        (self.site / "unregistered.md").write_text(
            "---\ntitle: Old page\ndescription: Preserved example.\n---\n"
            ":::caution[Pre-0.7 reference]\nOld content.\n:::\n"
        )
        self.assertIn("unclassified-page", self.rules())

    def test_unknown_example_is_discovered(self) -> None:
        (self.root / "examples/new.py").write_text("print(1)\n")
        self.assertIn("unclassified-example", self.rules())

    def test_missing_example_is_not_a_pass(self) -> None:
        self.example.unlink()
        self.assertIn("missing-example", self.rules())

    def test_empty_source_roots_fail(self) -> None:
        self.page.unlink()
        self.example.unlink()
        self.assertIn("empty-page-inventory", self.rules())
        self.assertIn("empty-example-inventory", self.rules())

    def test_heading_inside_code_does_not_satisfy_section(self) -> None:
        self.page.write_text(self.page_text().replace("## Limitations\n", "```text\n## Limitations\n") + "```\n")
        self.assertIn("section-limitations", self.rules())

    def test_empty_or_placeholder_section_fails(self) -> None:
        self.page.write_text(
            self.page_text().replace(
                "This synthetic calculation does not establish external empirical validity.", "TODO"
            )
        )
        self.assertIn("section-limitations", self.rules())
        self.assertIn("placeholder-content", self.rules())

    def test_mentioned_source_path_is_not_source_binding(self) -> None:
        self.page.write_text(
            self.page_text().replace('<Code code={forecastSource} lang="python" />', "examples/forecast.py")
        )
        self.assertIn("example-source-binding", self.rules())

    def test_raw_import_must_resolve_to_declared_example(self) -> None:
        self.page.write_text(self.page_text().replace("examples/forecast.py?raw", "examples/unrelated.py?raw"))
        self.assertIn("example-source-binding", self.rules())

    def test_raw_import_does_not_cover_unrelated_python_blocks(self) -> None:
        self.page.write_text(self.page_text() + "\n```python\nprint(undefined_variable)\n```\n")
        self.assertIn("unbound-python-example", self.rules())

    def test_reference_can_bind_multiple_complete_source_fences(self) -> None:
        second = self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"] = {
            "kind": "reference",
            "examples": ["examples/forecast.py", "examples/second.py"],
        }
        self.page.write_text(
            "---\ntitle: Checked references\ndescription: Two separately declared complete Python examples.\n---\n"
            f"```python\n{self.example.read_text()}```\n"
            f"```python\n{second.read_text()}```\n"
        )
        report = self.report()
        self.assertEqual([], report["violations"])
        self.assertEqual(0, report["counts"]["unbound_python_blocks"])

    def test_tutorial_keeps_primary_and_allows_an_additional_source(self) -> None:
        second = self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        self.page.write_text(self.page_text() + f"\n```python\n{second.read_text()}```\n")
        self.assertEqual([], self.report()["violations"])

    def test_multiple_raw_import_components_bind_their_declared_sources(self) -> None:
        self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        self.page.write_text(
            self.page_text()
            + "\nimport secondSource from '../../../../../../examples/second.py?raw';\n"
            + '<Code code={secondSource} lang="python" />\n'
        )
        self.assertEqual([], self.report()["violations"])

    def test_registered_but_undeclared_third_source_is_not_covered(self) -> None:
        second = self.register_example("second.py", 8)
        third = self.register_example("third.py", 13)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        self.page.write_text(
            self.page_text() + f"\n```python\n{second.read_text()}```\n```python\n{third.read_text()}```\n"
        )
        report = self.report()
        self.assertEqual(1, report["counts"]["unbound_python_blocks"])
        self.assertEqual(["unbound-python-example"], [item["rule"] for item in report["violations"]])

    def test_additional_declared_source_must_actually_be_displayed(self) -> None:
        self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        violations = self.report()["violations"]
        self.assertEqual(["example-source-binding"], [item["rule"] for item in violations])
        self.assertIn("examples/second.py", violations[0]["detail"])

    def test_commenting_an_additional_source_does_not_count_as_display(self) -> None:
        second = self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        self.page.write_text(self.page_text() + f"\n{{/*\n```python\n{second.read_text()}```\n*/}}\n")
        self.assertIn("example-source-binding", self.rules())

    def test_missing_additional_source_file_is_rejected(self) -> None:
        second = self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = ["examples/second.py"]
        second.unlink()
        self.assertTrue({"missing-example", "example-source-binding"} <= self.rules())

    def test_additional_examples_do_not_replace_a_tutorials_primary_example(self) -> None:
        entry = self.contract["pages"]["user-guide/quickstart.mdx"]
        entry.pop("example")
        entry["examples"] = ["examples/forecast.py"]
        self.assertEqual({"tutorial-example"}, self.rules())

    def test_additional_example_list_has_a_strict_schema(self) -> None:
        self.register_example("second.py", 8)
        for invalid in (
            None,
            True,
            "examples/second.py",
            [],
            [42],
            [{}],
            ["../elsewhere.py"],
            ["examples/unknown.py"],
            ["examples/second.py", "examples/second.py"],
            ["examples/forecast.py"],
        ):
            with self.subTest(examples=invalid):
                self.contract["pages"]["user-guide/quickstart.mdx"]["examples"] = invalid
                self.assertEqual({"invalid-contract"}, self.rules())

    def test_historical_classification_does_not_hide_an_explicit_source_declaration(self) -> None:
        self.register_example("second.py", 8)
        self.contract["pages"]["user-guide/quickstart.mdx"] = {
            "kind": "historical",
            "reason": "An earlier interface retained with an explicit example declaration.",
            "examples": ["examples/second.py"],
        }
        self.page.write_text(self.page_text() + "\n:::caution[Pre-0.7 reference]\nHistorical interface.\n:::\n")
        self.assertEqual({"example-source-binding"}, self.rules())

    def test_fenced_raw_import_does_not_count_as_live_binding(self) -> None:
        self.page.write_text(
            self.page_text().replace(
                "import forecastSource from '../../../../../../examples/forecast.py?raw';",
                "```mdx\nimport forecastSource from '../../../../../../examples/forecast.py?raw';\n```",
            )
        )
        self.assertIn("example-source-binding", self.rules())

    def test_commented_raw_import_and_component_do_not_count_as_binding(self) -> None:
        for opener, closer in [("{/*", "*/}"), ("<!--", "-->"), ("{ /*", "*/ }")]:
            with self.subTest(comment=opener):
                self.page.write_text(
                    self.page_text()
                    .replace("import { Code }", f"{opener}\nimport {{ Code }}", 1)
                    .replace(
                        '<Code code={forecastSource} lang="python" />',
                        f'<Code code={{forecastSource}} lang="python" />\n{closer}',
                        1,
                    )
                )
                self.assertIn("example-source-binding", self.rules())

    def test_commented_historical_notice_is_not_visible(self) -> None:
        self.contract["pages"]["user-guide/quickstart.mdx"] = {"kind": "historical", "reason": "Old API retained."}
        for opener, closer in [("{/*", "*/}"), ("<!--", "-->")]:
            with self.subTest(comment=opener):
                self.page.write_text(self.page_text() + f"\n{opener}\n:::caution[Pre-0.7 reference]\n{closer}\n")
                self.assertIn("historical-notice", self.rules())

    def test_commented_tutorial_heading_does_not_satisfy_section(self) -> None:
        self.page.write_text(self.page_text().replace("## Limitations\n", "{/*\n## Limitations\n") + "\n*/}\n")
        self.assertIn("section-limitations", self.rules())

    def test_comment_markers_inside_code_are_preserved(self) -> None:
        code = 'value = "<!-- literal -->"\nassert value.startswith("<!")\nprint("checked")'
        self.example.write_text(code + "\n")
        self.page.write_text(
            self.page_text().replace('<Code code={forecastSource} lang="python" />', f"```python\n{code}\n```")
        )
        self.assertNotIn("example-source-binding", self.rules())
        self.assertNotIn("unbound-python-example", self.rules())

    def test_trivial_assertion_does_not_count(self) -> None:
        self.example.write_text('assert True\nprint("checked")\n')
        self.assertIn("example-assertion", self.rules())

    def test_constant_equality_assertion_does_not_count(self) -> None:
        self.example.write_text('assert 1 == 1\nprint("checked")\n')
        self.assertIn("example-assertion", self.rules())

    def test_archived_version_is_counted_but_not_current_evidence(self) -> None:
        archive = self.site / "latest/guide.md"
        archive.parent.mkdir()
        archive.write_text("Historical reference without current execution proof.\n")
        report = self.report()
        self.assertFalse(report["complete"])
        self.assertTrue(report["current_structure_complete"])
        self.assertFalse(report["full_scope_complete"])
        self.assertEqual(1, report["counts"]["versioned_archive_pages"])
        self.assertEqual("historical_version", report["pages"][0]["classification"])

    def test_historical_classification_needs_visible_notice(self) -> None:
        self.contract["pages"]["user-guide/quickstart.mdx"] = {"kind": "historical", "reason": "Old API retained."}
        self.assertIn("historical-notice", self.rules())

    def test_bad_manifest_and_path_escape_fail_closed(self) -> None:
        self.contract["pages"]["../../outside.md"] = {"kind": "tutorial"}
        self.contract["examples"]["examples/forecast.py"]["network"] = "false"
        self.assertIn("invalid-contract", self.rules())

    def test_ratchet_detects_modified_debt_and_never_claims_complete(self) -> None:
        self.page.write_text(self.page_text().replace("## Limitations", "## Caveats"))
        before = self.report()
        baseline = checker.make_baseline(before)
        evaluated = checker.evaluate_ratchet(before, baseline)
        self.assertTrue(evaluated["passed"])
        self.assertFalse(before["complete"])
        self.page.write_text(self.page.read_text() + "\nNew unsupported statement.\n")
        self.assertFalse(checker.evaluate_ratchet(self.report(), baseline)["passed"])

    def test_ratchet_rejects_changed_policy_or_malformed_baseline(self) -> None:
        before = self.report()
        baseline = checker.make_baseline(before)
        self.contract["pages"]["user-guide/quickstart.mdx"]["kind"] = "reference"
        self.assertFalse(checker.evaluate_ratchet(self.report(), baseline)["passed"])
        self.assertFalse(checker.evaluate_ratchet(before, {"schema_version": 1})["passed"])

    def test_ratchet_rejects_resolved_baseline_entries(self) -> None:
        self.page.write_text(self.page_text().replace("## Limitations", "## Caveats"))
        baseline = checker.make_baseline(self.report())
        self.page.write_text(self.page_text())
        result = checker.evaluate_ratchet(self.report(), baseline)
        self.assertFalse(result["passed"])
        self.assertEqual(1, len(result["stale"]))

    def test_ratchet_rejects_duplicate_baseline_entries(self) -> None:
        self.page.write_text(self.page_text().replace("## Limitations", "## Caveats"))
        report = self.report()
        baseline = checker.make_baseline(report)
        baseline["violations"] = baseline["violations"] * 2
        self.assertFalse(checker.evaluate_ratchet(report, baseline)["passed"])

    def test_generated_api_absence_is_a_hard_failure(self) -> None:
        self.contract["generated_api"] = dict(checker.GENERATED_API)
        report = self.report()
        self.assertEqual("not_built_or_invalid", report["generated_api_status"])
        self.assertIn("generated-api-evidence", self.rules())
        self.assertFalse(checker.evaluate_ratchet(report, checker.make_baseline(report))["passed"])

    def test_generated_api_policy_cannot_select_arbitrary_code(self) -> None:
        self.contract["generated_api"] = {"generator": "../../elsewhere.py", "manifest": "bypass.json"}
        self.assertIn("invalid-contract", self.rules())

    def test_generated_pages_require_recomputed_content_match(self) -> None:
        self.contract["generated_api"] = dict(checker.GENERATED_API)
        generated = self.site / "api/generated"
        generated.mkdir(parents=True)
        page = generated / "index.md"
        content = "---\ntitle: API reference\ndescription: Generated documented symbols.\n---\nReference text.\n"
        page.write_text(content)
        (generated / "manifest.json").write_text(json.dumps({"pages": {"index.md": {"symbols": []}}}))
        scripts = self.root / "scripts"
        scripts.mkdir()
        relative = page.relative_to(self.root).as_posix()
        (scripts / "generate_api_reference.py").write_text(
            f"CONTENT = {content!r}\nPATH = {relative!r}\n"
            "def build_artifacts(root):\n    return {PATH: CONTENT}\n"
            "def check_generated(root):\n"
            "    return [] if (root / PATH).read_text() == CONTENT else ['Stale generated content']\n"
        )
        report = self.report()
        self.assertEqual("verified", report["generated_api_status"])
        self.assertEqual(1, report["counts"]["generated_api_pages"])
        self.assertEqual([], report["violations"])
        page.write_text(content + "Unqualified addition.\n")
        self.assertIn("generated-api-evidence", self.rules())
        self.assertIn("unclassified-page", self.rules())

    def test_generated_directory_is_not_an_exemption(self) -> None:
        generated = self.site / "api/generated"
        generated.mkdir(parents=True)
        (generated / "invented.md").write_text("---\ntitle: Fake\ndescription: Unverified API page.\n---\n")
        self.assertIn("unclassified-page", self.rules())

    def doctest_fixture(self, source_text: str, installed_text: str | None = None) -> str:
        """Create an actual isolated installed module without a native dependency."""
        source = self.root / "bindings/python/innovate/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text(source_text)
        generated = self.site / "api/generated"
        generated.mkdir(parents=True)
        page = generated / "index.md"
        content = "---\ntitle: API reference\ndescription: Exact source test fixture.\n---\n"
        page.write_text(content)
        (generated / "manifest.json").write_text(
            json.dumps(
                {
                    "pages": {"index.md": {"symbols": []}},
                    "sources": {
                        "bindings/python/innovate/__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()
                    },
                }
            )
        )
        scripts = self.root / "scripts"
        scripts.mkdir()
        relative = page.relative_to(self.root).as_posix()
        (scripts / "generate_api_reference.py").write_text(
            f"CONTENT = {content!r}\nPATH = {relative!r}\n"
            "def build_artifacts(root):\n    return {PATH: CONTENT}\n"
            "def check_generated(root):\n    return []\n"
        )
        environment = self.root / "installed"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        python = str(environment / "bin/python")
        site_packages = Path(
            subprocess.check_output(
                [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
            ).strip()
        )
        package = site_packages / "innovate"
        package.mkdir()
        (package / "__init__.py").write_text(source_text if installed_text is None else installed_text)
        return python

    def test_api_doctest_execution_binds_installed_source_and_attempts(self) -> None:
        python = self.doctest_fixture('"""A synthetic doctest.\n\n>>> 2 + 2\n4\n"""\n')
        result = checker.execute_api_doctests(self.root, python)[0]
        self.assertEqual("passed", result["status"])
        self.assertEqual(1, result["attempted"])
        self.assertEqual(result["source_sha256"], result["installed_source_sha256"])
        self.assertEqual("3.15.2", result["python_version"])

    def test_api_doctests_reject_stale_installed_binding(self) -> None:
        python = self.doctest_fixture('"""A synthetic doctest.\n\n>>> 2 + 2\n4\n"""\n', '"""Stale wheel."""\n')
        result = checker.execute_api_doctests(self.root, python)[0]
        self.assertEqual("failed", result["status"])
        self.assertIn("Installed binding source differs", result["stderr"])

    def test_api_doctests_reject_skipped_examples(self) -> None:
        python = self.doctest_fixture('"""A synthetic doctest.\n\n>>> 2 + 2  # doctest: +SKIP\n4\n"""\n')
        result = checker.execute_api_doctests(self.root, python)[0]
        self.assertEqual("failed", result["status"])
        self.assertIn("Skipped docstring examples", result["stderr"])

    def test_cli_preserves_virtual_environment_interpreter_path(self) -> None:
        python = self.doctest_fixture('"""Installed fixture.\n\n>>> value\n4\n"""\nvalue = 4\n')
        self.example.write_text('import innovate\nassert innovate.value == 4\nprint("checked")\n')
        self.contract["generated_api"] = dict(checker.GENERATED_API)
        contract_path = self.root / checker.CONTRACT
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(self.contract))
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(["--root", str(self.root), "--mode", "audit", "--execute", "--python", python])
        self.assertEqual(0, status)
        report = json.loads(output.getvalue())
        self.assertEqual("passed", report["execution_status"])
        self.assertEqual(2, len(report["executions"]))

    def test_strict_cli_requires_execution_when_structure_has_no_gaps(self) -> None:
        contract_path = self.root / checker.CONTRACT
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(self.contract))
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(["--root", str(self.root), "--mode", "strict"])
        self.assertEqual(1, status)
        report = json.loads(output.getvalue())
        self.assertTrue(report["current_structure_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual("not_run", report["execution_status"])

    def test_strict_cli_passes_only_with_examples_and_doctests_executed(self) -> None:
        python = self.doctest_fixture('"""Installed fixture.\n\n>>> value\n4\n"""\nvalue = 4\n')
        self.contract["generated_api"] = dict(checker.GENERATED_API)
        contract_path = self.root / checker.CONTRACT
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(self.contract))
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(["--root", str(self.root), "--mode", "strict", "--execute", "--python", python])
        self.assertEqual(0, status)
        report = json.loads(output.getvalue())
        self.assertTrue(report["complete"])
        self.assertTrue(report["current_structure_complete"])
        self.assertTrue(report["current_content_complete"])
        self.assertEqual("passed", report["api_doctest_status"])

    def test_executed_examples_without_api_doctests_cannot_pass_strict(self) -> None:
        contract_path = self.root / checker.CONTRACT
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(self.contract))
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(
                ["--root", str(self.root), "--mode", "strict", "--execute", "--python", sys.executable]
            )
        self.assertEqual(1, status)
        report = json.loads(output.getvalue())
        self.assertFalse(report["complete"])

    def test_execution_runs_real_example_in_isolation(self) -> None:
        result = checker.execute_examples(self.root, self.contract, sys.executable)
        self.assertEqual("passed", result[0]["status"])
        self.assertEqual("checked\n", result[0]["stdout"])
        self.assertEqual(64, len(result[0]["source_sha256"]))

    def test_failed_assertion_is_failed_execution(self) -> None:
        self.example.write_text("value = 2\nassert value == 3\n")
        result = checker.execute_examples(self.root, self.contract, sys.executable)
        self.assertEqual("failed", result[0]["status"])
        self.assertNotEqual(0, result[0]["returncode"])

    def test_unreachable_assertion_does_not_pass_execution(self) -> None:
        self.example.write_text("def never_called():\n    value = 2\n    assert value == 3\n")
        result = checker.execute_examples(self.root, self.contract, sys.executable)
        self.assertEqual("failed", result[0]["status"])
        self.assertEqual([], result[0]["executed_assertion_lines"])

    def test_wrong_documented_output_fails_executed_example(self) -> None:
        self.contract["examples"]["examples/forecast.py"]["expected_stdout"] = "not the real output\n"
        result = checker.execute_examples(self.root, self.contract, sys.executable)
        self.assertEqual("failed", result[0]["status"])
        self.assertFalse(result[0]["stdout_matches_contract"])

    def test_missing_interpreter_is_unexecuted(self) -> None:
        result = checker.execute_examples(self.root, self.contract, str(self.root / "missing-python"))
        self.assertEqual("not_run", result[0]["status"])

    def native_receipt_fixture(self) -> Path:
        """Build synthetic coherent receipts to test aggregation, not native execution."""
        scripts = self.root / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "check_polyglot_api_documentation.py").write_text("def check_inventory(root):\n    return []\n")
        source_paths = {
            "bindings/julia/src/Innovate.jl": "# synthetic Julia source\r\n",
            "bindings/r/R/kernel_bridge.R": "# synthetic R source\n",
            "bindings/r/man/example.Rd": "Synthetic manual\n",
            "scripts/check_polyglot_api_documentation.py": "def check_inventory(root):\n    return []\n",
            "scripts/check_julia_documentation.jl": "# synthetic Julia runner\n",
            "scripts/check_r_documentation.R": "# synthetic R runner\n",
        }
        sources = {}
        for relative, content in source_paths.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
            sources[relative] = hashlib.sha256(content.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()
        inventory = {
            "schema_version": 1,
            "sources": sources,
            "symbols": {language: [{"name": "example"}] for language in ("julia", "r")},
            "native_examples": {},
            "counts": {},
        }
        native = self.root / "target/release/libinnovate_ffi.so"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"synthetic native identity")
        receipts = self.root / "receipts"
        receipts.mkdir()
        for language, source in (("julia", "bindings/julia/src/Innovate.jl"), ("r", "bindings/r/man/example.Rd")):
            code = "@assert 2 + 2 == 4\n" if language == "julia" else "stopifnot(2 + 2 == 4)\n"
            inventory["native_examples"][language] = [
                {
                    "id": f"{language}:example:{number}",
                    "source": source,
                    "source_line": number,
                    "source_sha256": sources[source],
                    "code": code,
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "assertion_count": 1,
                }
                for number in (1, 2)
            ]
            inventory["counts"].update({f"{language}_example_blocks": 2, f"{language}_example_assertions": 2})
        inventory_path = self.root / "docs/contracts/polyglot-api-documentation.json"
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
        inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        for language in ("julia", "r"):
            receipt = {
                "schema_version": 1,
                "language": language,
                "status": "passed",
                "qualified": True,
                "runtime": "1.13.0" if language == "julia" else "R version 4.7.0",
                "platform": "synthetic-test-platform",
                "package_version": "0.7.0",
                "runner": f"scripts/check_{language}_documentation." + ("jl" if language == "julia" else "R"),
                "inventory_sha256": inventory_sha,
                "source_sha256": sources,
                "exported_names_checked": ["example"],
                "examples_executed": 2,
                "assertions_executed": 2,
                "native_library": {
                    "path": "/synthetic/libinnovate_ffi.so",
                    "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                },
                "examples": [
                    {key: item[key] for key in ("id", "source", "source_line", "code_sha256")}
                    | {"status": "passed", "assertions_executed": 1}
                    for item in inventory["native_examples"][language]
                ],
            }
            if language == "julia":
                receipt.update({"docstrings_checked": 1, "json_version": "1.13.4"})
            else:
                receipt.update(
                    {
                        "jsonlite_version": "2.0.0",
                        "manuals_checked": ["bindings/r/man/example.Rd"],
                        "installed_code_sha256": {
                            "/synthetic/innovate.R.rdb": "a" * 64,
                            "/synthetic/innovate.R.rdx": "b" * 64,
                        },
                    }
                )
            (receipts / f"{language}-documentation.json").write_text(json.dumps(receipt))
        return receipts

    def test_language_receipts_require_every_declared_language(self) -> None:
        receipts = self.native_receipt_fixture()
        (receipts / "r-documentation.json").unlink()
        result = checker.verify_language_receipts(self.root, receipts, ["python", "julia", "r"])
        self.assertEqual("passed", result["julia"]["status"])
        self.assertNotEqual("passed", result["r"]["status"])

    def test_complete_source_bound_language_receipts_pass(self) -> None:
        receipts = self.native_receipt_fixture()
        result = checker.verify_language_receipts(self.root, receipts, ["python", "julia", "r"])
        self.assertTrue(all(item["status"] == "passed" for item in result.values()))

    def test_stale_language_source_and_inventory_are_rejected(self) -> None:
        receipts = self.native_receipt_fixture()
        source = self.root / "bindings/julia/src/Innovate.jl"
        source.write_text(source.read_text() + "changed\n")
        result = checker.verify_language_receipts(self.root, receipts, ["python", "julia", "r"])
        self.assertTrue(all(item["status"] != "passed" for item in result.values()))

    def test_partial_duplicate_or_forged_native_examples_are_rejected(self) -> None:
        receipts = self.native_receipt_fixture()
        path = receipts / "julia-documentation.json"
        original = json.loads(path.read_text())
        for defect in ("partial", "duplicate", "hash", "assertions", "bool_assertions", "qualified"):
            with self.subTest(defect=defect):
                value = copy.deepcopy(original)
                if defect == "partial":
                    value["examples"].pop()
                    value["examples_executed"] = value["assertions_executed"] = 1
                elif defect == "duplicate":
                    value["examples"][1] = value["examples"][0]
                elif defect == "hash":
                    value["examples"][0]["code_sha256"] = "f" * 64
                elif defect in {"assertions", "bool_assertions"}:
                    value["examples"][0]["assertions_executed"] = 0 if defect == "assertions" else True
                else:
                    value["qualified"] = "true"
                path.write_text(json.dumps(value))
                result = checker.verify_language_receipts(self.root, receipts, ["python", "julia"])
                self.assertNotEqual("passed", result["julia"]["status"])

    def test_native_receipt_requires_runtime_and_current_binary_identity(self) -> None:
        receipts = self.native_receipt_fixture()
        path = receipts / "r-documentation.json"
        receipt = json.loads(path.read_text())
        receipt["runtime"] = ""
        path.write_text(json.dumps(receipt))
        result = checker.verify_language_receipts(self.root, receipts, ["python", "r"])
        self.assertNotEqual("passed", result["r"]["status"])
        (self.root / "target/release/libinnovate_ffi.so").write_bytes(b"different build")
        result = checker.verify_language_receipts(self.root, receipts, ["python", "julia"])
        self.assertNotEqual("passed", result["julia"]["status"])

    def test_python_execution_does_not_complete_required_julia_and_r_scope(self) -> None:
        python = self.doctest_fixture('"""Installed fixture.\n\n>>> value\n4\n"""\nvalue = 4\n')
        receipts = self.native_receipt_fixture()
        (self.root / "docs/contracts/api-documentation.json").write_text(
            json.dumps({"generated_reference_languages": ["python", "julia", "r"]})
        )
        self.contract["generated_api"] = dict(checker.GENERATED_API)
        (self.root / checker.CONTRACT).write_text(json.dumps(self.contract))
        output = io.StringIO()
        arguments = ["--root", str(self.root), "--mode", "strict", "--execute", "--python", python]
        with redirect_stdout(output):
            missing_status = checker.main(arguments)
        missing = json.loads(output.getvalue())
        self.assertEqual(1, missing_status)
        self.assertEqual("passed", missing["python_execution_status"])
        self.assertTrue(missing["current_python_content_complete"])
        self.assertFalse(missing["current_content_complete"])
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(arguments + ["--language-receipt-dir", str(receipts)])
        report = json.loads(output.getvalue())
        self.assertEqual(0, status)
        self.assertTrue(report["complete"])

    def test_unknown_or_empty_language_scope_fails_closed(self) -> None:
        policy = self.root / "docs/contracts/api-documentation.json"
        for languages in ([], ["python", "python"], ["python", "unknown"], "python", None):
            with self.subTest(languages=languages):
                policy.write_text(json.dumps({"generated_reference_languages": languages}))
                self.assertIn("invalid-language-contract", self.rules())

    def test_malformed_or_stale_receipt_metadata_is_rejected(self) -> None:
        receipts = self.native_receipt_fixture()
        path = receipts / "r-documentation.json"
        original = json.loads(path.read_text())
        mutations = {
            "inventory": {"inventory_sha256": "0" * 64},
            "partial_sources": {"source_sha256": {}},
            "status": {"status": "failed"},
            "schema": {"schema_version": True},
            "language": {"language": "julia"},
            "runner": {"runner": "scripts/unreviewed.R"},
            "exports": {"exported_names_checked": []},
            "manuals": {"manuals_checked": []},
            "installed_code": {"installed_code_sha256": {}},
            "total": {"assertions_executed": True},
            "example_row": {"examples": [None, None]},
        }
        for defect, changes in mutations.items():
            with self.subTest(defect=defect):
                path.write_text(json.dumps(original | changes))
                result = checker.verify_language_receipts(self.root, receipts, ["python", "r"])
                self.assertNotEqual("passed", result["r"]["status"])
        for malformed in ("{", "null", '"receipt"', '{"status":"failed","status":"passed"}'):
            with self.subTest(malformed=malformed):
                path.write_text(malformed)
                result = checker.verify_language_receipts(self.root, receipts, ["python", "r"])
                self.assertNotEqual("passed", result["r"]["status"])

    def test_structural_inventory_rejection_invalidates_native_receipts(self) -> None:
        receipts = self.native_receipt_fixture()
        path = self.root / "scripts/check_polyglot_api_documentation.py"
        path.write_text('def check_inventory(root):\n    return ["inventory differs from current source"]\n')
        result = checker.verify_language_receipts(self.root, receipts, ["python", "julia", "r"])
        self.assertTrue(all(item["status"] == "failed" for item in result.values()))

    def test_static_ratchet_cannot_claim_native_receipt_execution(self) -> None:
        receipts = self.native_receipt_fixture()
        (self.root / "docs/contracts/api-documentation.json").write_text(
            json.dumps({"generated_reference_languages": ["python", "julia", "r"]})
        )
        (self.root / checker.CONTRACT).write_text(json.dumps(self.contract))
        report = checker.build_report(self.root)
        (self.root / checker.BASELINE).write_text(json.dumps(checker.make_baseline(report)))
        output = io.StringIO()
        with redirect_stdout(output):
            status = checker.main(
                ["--root", str(self.root), "--mode", "ratchet", "--language-receipt-dir", str(receipts)]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(0, status)
        self.assertEqual("not_run", result["execution_status"])
        self.assertEqual("not_run", result["python_execution_status"])
        self.assertTrue(all(item["status"] == "not_run" for item in result["language_receipts"].values()))
        self.assertFalse(result["current_python_content_complete"])
        self.assertFalse(result["current_content_complete"])
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
