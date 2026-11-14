"""Keep hidden documentation receipts uploadable without widening the artifact scope.

These dependency-free, format-sensitive checks guard this repository's workflow;
they are not a YAML parser or a substitute for hosted artifact-upload execution.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"
BUILD_RECEIPTS = ("api.json", "polyglot-api.json", "content.json", "artifact.json")
EXECUTION_RECEIPTS = ("native-api.json", "julia-documentation.json", "r-documentation.json", "execution.json")


class DocumentationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def step(self, name):
        pattern = rf"^      - name: {re.escape(name)}\n(.*?)(?=^      - |^  [\w-]+:|\Z)"
        matches = re.findall(pattern, self.workflow, flags=re.MULTILINE | re.DOTALL)
        self.assertEqual(len(matches), 1, f"Expected exactly one step: {name}")
        return matches[0]

    def assert_receipt_upload(self, step, artifact, receipts):
        self.assertIn("        if: always()\n", step)
        self.assertRegex(step, r"(?m)^        uses: actions/upload-artifact@[0-9a-f]{40}$")
        self.assertIn(f"          name: {artifact}\n", step)
        self.assertIn("          include-hidden-files: true\n", step)
        self.assertIn("          if-no-files-found: error\n", step)
        paths = re.findall(r"(?m)^          path: \|\n((?:            [^\n]+\n)+)", step)
        self.assertEqual(len(paths), 1, "Evidence must use one explicit file allowlist")
        self.assertEqual(
            paths[0].split(),
            [f"docs/astro-site/.reports/{name}" for name in receipts],
            "Do not upload the workspace, the whole reports directory, or arbitrary JSON files",
        )
        self.assertNotIn("continue-on-error:", step)

    def test_build_evidence_upload_is_explicit_and_fail_closed(self):
        self.assert_receipt_upload(
            self.step("Retain documentation evidence"), "documentation-build-evidence", BUILD_RECEIPTS
        )

    def test_execution_evidence_upload_is_explicit_and_fail_closed(self):
        self.assert_receipt_upload(
            self.step("Retain executed-example evidence"), "documentation-execution-evidence", EXECUTION_RECEIPTS
        )

    def test_regression_suite_runs_before_the_build(self):
        test_step = self.step("Test documentation contracts")
        self.assertRegex(test_step, r"for suite in [^\n]*\btest_documentation_workflow\b")
        self.assertLess(
            self.workflow.index("- name: Test documentation contracts"),
            self.workflow.index("- name: Build documentation site"),
        )

    def test_deployment_keeps_both_qualification_jobs_and_explicit_activation(self):
        deployment = self.workflow.split("\n  deploy:\n", 1)[1]
        self.assertIn("    needs: [build, examples]\n", deployment)
        self.assertIn("vars.ENABLE_PAGES_ACTIONS_DEPLOY == 'true'", deployment)
        self.assertIn("github.ref == 'refs/heads/main'", deployment)
        self.assertNotIn("always()", deployment)

    def test_hidden_file_omission_and_empty_upload_relaxation_are_rejected(self):
        step = self.step("Retain documentation evidence")
        for original, replacement in (
            ("          include-hidden-files: true\n", ""),
            ("include-hidden-files: true", "include-hidden-files: false"),
            ("if-no-files-found: error", "if-no-files-found: warn"),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(AssertionError):
                    self.assert_receipt_upload(
                        step.replace(original, replacement), "documentation-build-evidence", BUILD_RECEIPTS
                    )

    def test_unrelated_hidden_files_and_wildcards_are_rejected(self):
        step = self.step("Retain documentation evidence")
        for filename in (".env", ".private.json", "*.json", "../.env", "../../.git/config"):
            with self.subTest(filename=filename):
                changed = step.replace(".reports/api.json", f".reports/{filename}")
                with self.assertRaises(AssertionError):
                    self.assert_receipt_upload(changed, "documentation-build-evidence", BUILD_RECEIPTS)


if __name__ == "__main__":
    unittest.main()
