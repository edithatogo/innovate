"""Regressions for stale review evidence and unsupported human-validation claims."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scientific_review_checker", ROOT / "scripts/check_scientific_review_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ScientificReviewContractTests(unittest.TestCase):
    """Use a fresh repository fixture rather than altering committed evidence."""

    def setUp(self) -> None:
        """Create one source-bound agent review with no human observations."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.page = "docs/astro-site/src/content/docs/review-fixture.mdx"
        self.text = "# Synthetic review fixture\n\nLevel = 50. Units: études.\nhttps://example.org/primary\n"
        source = self.root / self.page
        source.parent.mkdir(parents=True)
        source.write_bytes(self.text.replace("\n", "\r\n").encode("utf-8"))
        self.record = {
            "schema_version": 1,
            "review_date": "2026-11-15",
            "review_timezone": "Australia/Brisbane",
            "base_commit": "a" * 40,
            "reviewer": {"kind": "agent", "role": "synthetic test fixture", "human_reviewer": False},
            "status": "scoped_agent_review_complete",
            "scope": {
                "pages": [self.page],
                "native_guide_sections": ["Not used by this synthetic fixture"],
                "focus": ["Interpretation"],
                "exclusions": ["Human and scientific validation"],
            },
            "fingerprint_encoding": checker.ENCODING,
            "source_files": {self.page: hashlib.sha256(self.text.encode("utf-8")).hexdigest()},
            "references": [
                {
                    "id": "fixture-source",
                    "title": "Synthetic fixture reference",
                    "url": "https://example.org/primary",
                    "source_type": "synthetic test metadata; not a research claim",
                    "inspected_locations": ["Fixture only"],
                    "supports": "Exercises a source-linked record.",
                    "limits": "No real publication or scientific validation is claimed.",
                }
            ],
            "findings": [
                {
                    "id": "fixture-interpretation",
                    "severity": "medium",
                    "page": self.page,
                    "status": "resolved_in_prose",
                    "finding": "Quantity needed a unit.",
                    "resolution": "The synthetic page now gives one.",
                    "references": ["fixture-source"],
                }
            ],
            "reader_tasks": [
                {
                    "id": "fixture-reader",
                    "page": self.page,
                    "prompt": "Read the synthetic level.",
                    "expected_answer": "50.",
                    "answer_review": "agent_checked",
                    "human_execution": "not_run",
                }
            ],
            "verification": {
                key: "Synthetic fixture; no execution evidence."
                for key in (
                    "python_fences",
                    "raw_example_imports",
                    "independent_arithmetic",
                    "native_execution",
                    "astro_build",
                    "citation_review",
                )
            }
            | {"human_participants": 0, "human_task_trials": 0},
            "completion": dict(checker.COMPLETION),
            "remaining": ["Human trials and empirical validation remain unperformed."],
        }

    def run_gate(self, record: dict | None = None) -> tuple[int, dict]:
        """Run the public CLI path and read its named JSON artifact."""
        path = self.root / checker.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record if record is None else record), encoding="utf-8")
        output = self.root / "reports/scientific-review.json"
        with redirect_stdout(io.StringIO()):
            status = checker.main(["--root", str(self.root), "--output", str(output)])
        return status, json.loads(output.read_text(encoding="utf-8"))

    def test_fresh_utf8_crlf_record_passes_without_certifying_quality(self) -> None:
        """Accept the declared newline normalization while retaining claim limits."""
        status, report = self.run_gate()
        self.assertEqual(0, status, report["errors"])
        self.assertEqual("passed", report["status"])
        self.assertFalse(report["scientific_quality_certified"])
        self.assertFalse(report["pedagogical_quality_certified"])
        self.assertFalse(report["human_reader_validation_performed"])
        self.assertFalse(report["native_execution_performed"])

    def test_relative_output_uses_nested_invocation_directory(self) -> None:
        """Keep the named receipt beside the site when --root points elsewhere."""
        self.assertEqual(0, self.run_gate()[0])
        invocation = self.root / "docs/astro-site"
        invocation.mkdir(parents=True, exist_ok=True)
        with chdir(invocation), redirect_stdout(io.StringIO()):
            status = checker.main(
                [
                    "--root",
                    str(self.root),
                    "--output",
                    ".reports/scientific-review.json",
                ]
            )
        output = invocation / ".reports/scientific-review.json"
        self.assertEqual(0, status)
        self.assertEqual("passed", json.loads(output.read_text(encoding="utf-8"))["status"])
        self.assertFalse((self.root / ".reports/scientific-review.json").exists())

    def test_changed_reviewed_source_fails_with_nonzero_status(self) -> None:
        """Reject a changed scientific value even when the review itself is intact."""
        self.assertEqual([], checker.validate_review(self.root, self.record))
        (self.root / self.page).write_text(self.text.replace("Level = 50", "Level = 51"), encoding="utf-8")
        status, report = self.run_gate()
        self.assertEqual(1, status)
        self.assertEqual("failed", report["status"])
        self.assertTrue(any("Reviewed source changed:" in error for error in report["errors"]))

    def test_imported_raw_source_cannot_lose_its_fingerprint(self) -> None:
        """Keep a displayed raw example bound to the review after page rehashing."""
        example = "examples/fixture.py"
        code = "value = 50\n"
        source = self.root / example
        source.parent.mkdir(parents=True)
        source.write_text(code, encoding="utf-8")
        page_text = self.text + "import fixture from '../../../../../examples/fixture.py?raw';\n"
        (self.root / self.page).write_text(page_text, encoding="utf-8")
        self.record["source_files"][self.page] = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        self.record["source_files"][example] = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.assertEqual([], checker.validate_review(self.root, self.record))
        del self.record["source_files"][example]
        status, report = self.run_gate()
        self.assertEqual(1, status)
        self.assertTrue(any("import has no source fingerprint" in error for error in report["errors"]))

    def test_finding_cannot_name_an_unreviewed_reference(self) -> None:
        """Reject a claim link whose identifier has no inspected source record."""
        self.assertEqual([], checker.validate_review(self.root, self.record))
        self.record["findings"][0]["references"] = ["unreviewed-source"]
        status, report = self.run_gate()
        self.assertEqual(1, status)
        self.assertTrue(any("has an unknown reference" in error for error in report["errors"]))

    def test_duplicate_json_keys_cannot_replace_review_evidence(self) -> None:
        """Reject duplicate keys before a decoder can silently select one value."""
        self.assertEqual(0, self.run_gate()[0])
        path = self.root / checker.CONTRACT
        raw = path.read_text(encoding="utf-8").replace(
            '"schema_version": 1',
            '"schema_version": 0, "schema_version": 1',
            1,
        )
        path.write_text(raw, encoding="utf-8")
        output = self.root / "reports/duplicate-key.json"
        with redirect_stdout(io.StringIO()):
            status = checker.main(["--root", str(self.root), "--output", str(output)])
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(1, status)
        self.assertTrue(any("Duplicate JSON key: schema_version" in error for error in report["errors"]))

    def test_human_validation_without_observations_is_rejected(self) -> None:
        """Prevent a true completion flag from laundering unexecuted reader tasks."""
        self.assertEqual([], checker.validate_review(self.root, self.record))
        forged = copy.deepcopy(self.record)
        forged["completion"]["human_reader_validation"] = True
        status, report = self.run_gate(forged)
        self.assertEqual(1, status)
        self.assertTrue(any("completion.human_reader_validation" in error for error in report["errors"]))
        self.assertFalse(report["human_reader_validation_performed"])


if __name__ == "__main__":
    unittest.main()
