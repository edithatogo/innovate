"""Exercise standards traceability without third-party or native dependencies."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_community_documentation.py"
CONTRACT = "docs/contracts/community-documentation.json"
PAGE = "docs/astro-site/src/content/docs/operations/documentation-standards.md"


class CommunityDocumentationTests(unittest.TestCase):
    """Keep mapping checks separate from runtime, release and provider acceptance."""

    def setUp(self):
        """Copy actual public references and registry evidence into an isolated fixture."""
        spec = importlib.util.spec_from_file_location("community_documentation", SCRIPT)
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)
        self.packet = json.loads((ROOT / CONTRACT).read_text(encoding="utf-8"))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        paths = {CONTRACT, PAGE, self.packet["history"]["path"], *self.packet["registry"]["paths"].values()}
        for requirement in self.packet["requirements"]:
            paths.update(requirement["evidence"])
            paths.update(requirement["checks"])
        registry = json.loads((ROOT / self.packet["registry"]["paths"]["manifest"]).read_text(encoding="utf-8"))
        paths.update(registry["input_sha256"])
        for destination in registry["destinations"]:
            paths.add(destination["dossier"])
        for relative in paths:
            destination = self.repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def validate(self, packet=None):
        """Validate a selected mutation or the unmodified fixture contract."""
        return self.checker.validate(self.repository, self.packet if packet is None else packet)

    def test_mapping_pass_does_not_claim_acceptance(self):
        report = self.validate()
        self.assertTrue(report["record_valid"])
        self.assertEqual(report["assessment"], "documentation_traceability_only")
        for field in (
            "online_verification_performed",
            "native_execution_performed",
            "external_acceptance_assessed",
            "scientific_validation_assessed",
            "release_qualification_assessed",
        ):
            self.assertIs(report[field], False)
        self.assertEqual(report["registry"]["pending_issues"], [399, 400, 401])
        self.assertFalse(report["registry"]["recorded_complete"])
        self.assertNotIn("complete", report)
        self.assertTrue(report["checks_not_run"])
        self.assertTrue(report["observed_source_sha256"])

    def test_rendered_page_is_current(self):
        report = self.checker.check(self.repository)
        self.assertTrue(report["page_matches_mapping"])
        self.assertEqual(
            (self.repository / PAGE).read_text(encoding="utf-8"), self.checker.render_page(self.packet, report)
        )

    def test_invalid_identity_and_review_metadata_fail(self):
        cases = [
            ("schema_version", True),
            ("schema_version", 2),
            ("reviewed_by", "human"),
            ("reviewed_on", "tomorrow"),
            ("reviewed_on", "2999-01-01"),
            ("repository", "https://github.com/another/project"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                packet = copy.deepcopy(self.packet)
                packet[key] = value
                with self.assertRaises(ValueError):
                    self.validate(packet)

    def test_requirement_identity_and_claims_fail_closed(self):
        for mutation in ("missing", "duplicate", "unknown", "status", "relationship", "source"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.packet)
                requirement = packet["requirements"][0]
                if mutation == "missing":
                    packet["requirements"].pop()
                elif mutation == "duplicate":
                    packet["requirements"].append(copy.deepcopy(requirement))
                elif mutation == "unknown":
                    requirement["id"] = "invented-certification"
                elif mutation == "status":
                    requirement["status"] = "accepted"
                elif mutation == "relationship":
                    requirement["relationship"] = "historical_aspiration"
                else:
                    requirement["source_ids"] = ["numfocus_projects"]
                with self.assertRaises(ValueError):
                    self.validate(packet)

    def test_unsupported_claim_fields_fail(self):
        for field in ("certified", "external_acceptance", "scientifically_validated", "complete"):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet[field] = True
                with self.assertRaisesRegex(ValueError, "fields"):
                    self.validate(packet)

    def test_changed_official_source_is_not_trusted(self):
        urls = [
            "http://joss.readthedocs.io/en/latest/review_criteria.html",
            "https://joss.readthedocs.io.evil.example/en/latest/review_criteria.html",
            "https://joss.readthedocs.io/en/latest/review_criteria.html?accepted=true",
            "https://example.com/review",
        ]
        for url in urls:
            with self.subTest(url=url):
                packet = copy.deepcopy(self.packet)
                packet["sources"][0]["url"] = url
                with self.assertRaisesRegex(ValueError, "source"):
                    self.validate(packet)

    def test_source_date_and_access_limit_are_preserved(self):
        self.packet["sources"][0]["checked_on"] = "2020-01-01"
        with self.assertRaisesRegex(ValueError, "date"):
            self.validate()
        self.packet["sources"][0]["checked_on"] = self.packet["reviewed_on"]
        source = next(item for item in self.packet["sources"] if item["id"] == "scicrunch_registry")
        source["access"] = "official_page"
        with self.assertRaisesRegex(ValueError, "access"):
            self.validate()

    def test_noncanonical_evidence_paths_fail(self):
        for relative in ("../outside.md", "/etc/passwd", "README.md/../LICENSE", "./README.md"):
            with self.subTest(path=relative):
                packet = copy.deepcopy(self.packet)
                packet["requirements"][0]["evidence"].append(relative)
                with self.assertRaisesRegex(ValueError, "path"):
                    self.validate(packet)

    def test_symlink_escape_fails(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        outside = Path(temporary.name) / "public.md"
        outside.write_text("Public material outside the checked repository.\n", encoding="utf-8")
        (self.repository / "escape.md").symlink_to(outside)
        self.packet["requirements"][0]["evidence"].append("escape.md")
        with self.assertRaisesRegex(ValueError, "repository"):
            self.validate()

    def test_missing_and_empty_evidence_fail(self):
        path = self.repository / "README.md"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.validate()
        path.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Empty"):
            self.validate()

    def test_required_evidence_cannot_be_replaced_by_generic_file(self):
        requirement = next(item for item in self.packet["requirements"] if item["id"] == "joss_documentation")
        requirement["evidence"] = ["README.md"]
        with self.assertRaisesRegex(ValueError, "required evidence"):
            self.validate()

    def test_pending_work_cannot_be_erased(self):
        self.packet["requirements"][0]["remaining"] = []
        with self.assertRaisesRegex(ValueError, "remaining"):
            self.validate()

    def test_registry_issue_mapping_is_not_reassigned(self):
        self.packet["registry"]["issues"]["joss"] = 399
        with self.assertRaisesRegex(ValueError, "issue"):
            self.validate()

    def test_registry_validator_rejects_fabricated_outcome(self):
        manifest = self.repository / self.packet["registry"]["paths"]["manifest"]
        registry = json.loads(manifest.read_text(encoding="utf-8"))
        registry["destinations"][0].update(state="verified", outstanding=[], identifier="swh:1:rev:" + "a" * 40)
        manifest.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.validate()

    def test_registry_input_drift_still_fails(self):
        with (self.repository / "CITATION.cff").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed source\n")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.validate()

    def test_alternate_root_cannot_misidentify_the_executed_registry_policy(self):
        path = self.repository / self.packet["registry"]["paths"]["checker"]
        path.write_text("raise RuntimeError('never execute mapped evidence')\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "policy source"):
            self.validate()

    def test_historical_target_cannot_be_promoted(self):
        self.packet["history"]["status"] = "ready_for_review"
        with self.assertRaisesRegex(ValueError, "historical"):
            self.validate()

    def test_stale_or_invented_public_claims_fail(self):
        with (self.repository / PAGE).open("a", encoding="utf-8") as stream:
            stream.write("\nInnovate is certified by every listed community.\n")
        with self.assertRaisesRegex(ValueError, "page differs"):
            self.checker.check(self.repository)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.checker.read_json('{"schema_version": 1, "schema_version": 1}')

    def test_cli_reports_invalid_record_and_nonzero_exit(self):
        (self.repository / PAGE).write_text("Stale standards page.\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.repository), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        report = json.loads(result.stdout)
        self.assertFalse(report["record_valid"])
        self.assertFalse(report["online_verification_performed"])

    def test_explicit_regeneration_removes_stale_page_claims(self):
        (self.repository / PAGE).write_text("Unsupported acceptance claim.\n", encoding="utf-8")
        report = self.checker.check(self.repository, write_page=True)
        self.assertTrue(report["page_matches_mapping"])
        self.assertFalse(report["external_acceptance_assessed"])
        self.assertNotIn("Unsupported acceptance claim", (self.repository / PAGE).read_text(encoding="utf-8"))

    def test_regeneration_rejects_a_page_symlink_outside_repository(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        outside = Path(temporary.name) / "page.md"
        outside.write_text("Preserve this unrelated content.\n", encoding="utf-8")
        (self.repository / PAGE).unlink()
        (self.repository / PAGE).symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.checker.check(self.repository, write_page=True)
        self.assertEqual(outside.read_text(encoding="utf-8"), "Preserve this unrelated content.\n")

    def test_cli_success_output_stays_scoped(self):
        for json_output in (False, True):
            with self.subTest(json_output=json_output):
                arguments = ["--root", str(self.repository)] + (["--json"] if json_output else [])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(self.checker.main(arguments), 0)
                if json_output:
                    self.assertFalse(json.loads(output.getvalue())["external_acceptance_assessed"])
                else:
                    self.assertIn("external acceptance remain separate", output.getvalue())

    def test_cli_failure_output_does_not_look_like_a_pass(self):
        (self.repository / PAGE).write_text("Stale standards page.\n", encoding="utf-8")
        for json_output in (False, True):
            with self.subTest(json_output=json_output):
                arguments = ["--root", str(self.repository)] + (["--json"] if json_output else [])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(self.checker.main(arguments), 2)
                if json_output:
                    self.assertFalse(json.loads(output.getvalue())["record_valid"])
                else:
                    self.assertIn("Invalid documentation standards mapping", output.getvalue())


if __name__ == "__main__":
    unittest.main()
