"""Independent negative and positive controls for the CodeQL evidence verifier."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_codeql_success.py"
SHA = "a" * 40
PR = 42
REF = f"refs/pull/{PR}/head"
PATH = "dynamic/github-code-scanning/codeql"
SARIFS = ("12345678-1234-1234-1234-123456789abc", "87654321-4321-4321-4321-cba987654321")


def evidence():
    return {
        "pr": {"state": "open", "head": {"sha": SHA}, "base": {"ref": "main"}},
        "workflows": [{"id": 9, "path": PATH, "state": "active"}],
        "runs": [
            {
                "id": 100,
                "workflow_id": 9,
                "path": PATH,
                "event": "dynamic",
                "head_sha": SHA,
                "head_branch": REF,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "created_at": "2026-11-18T03:00:00Z",
                "run_started_at": "2026-11-18T03:00:00Z",
                "updated_at": "2026-11-18T03:05:00Z",
            }
        ],
        "jobs": [
            {
                "name": f"Analyze ({language})",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "run_id": 100,
            }
            for language in ("python", "rust")
        ],
        "analyses": [
            {
                "category": f"/language:{language}",
                "ref": REF,
                "commit_sha": SHA,
                "tool": {"name": "CodeQL"},
                "analysis_key": f"{PATH}:analyze",
                "created_at": "2026-11-18T03:01:00Z",
                "rules_count": 10,
                "sarif_id": SARIFS[index],
            }
            for index, language in enumerate(("python", "rust"))
        ],
        "aggregate": [
            {
                "id": 456,
                "name": "CodeQL",
                "head_sha": SHA,
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-11-18T03:01:00Z",
                "app": {"id": 57789},
                "pull_requests": [{"number": PR, "head": {"sha": SHA}, "base": {"ref": "main"}}],
            }
        ],
        "sarifs": {sarif: {"processing_status": "complete", "errors": None} for sarif in SARIFS},
        "alerts": [],
    }


class FakeApi:
    def __init__(self, value):
        self.value = value
        self.pr_reads = 0
        self.run_reads = 0
        self.after_pr = None
        self.after_runs = None

    def get(self, path, params=None):
        if path == f"/pulls/{PR}":
            self.pr_reads += 1
            return copy.deepcopy(self.after_pr if self.pr_reads > 1 and self.after_pr is not None else self.value["pr"])
        if path.startswith("/code-scanning/sarifs/"):
            return copy.deepcopy(self.value["sarifs"][path.rsplit("/", 1)[1]])
        raise AssertionError(f"unexpected request: {path}")

    def items(self, path, key=None, params=None):
        if path == "/actions/workflows":
            return copy.deepcopy(self.value["workflows"])
        if path == "/actions/workflows/9/runs":
            self.run_reads += 1
            return copy.deepcopy(
                self.after_runs if self.run_reads > 1 and self.after_runs is not None else self.value["runs"]
            )
        if path in {"/actions/runs/100/jobs", "/actions/runs/101/jobs"}:
            jobs = copy.deepcopy(self.value["jobs"])
            if path == "/actions/runs/101/jobs":
                for job in jobs:
                    job["run_id"] = 101
            return jobs
        if path == "/code-scanning/analyses":
            return copy.deepcopy(self.value["analyses"])
        if path == "/code-scanning/alerts":
            return copy.deepcopy(self.value["alerts"])
        if path == f"/commits/{SHA}/check-runs":
            return copy.deepcopy(self.value["aggregate"])
        raise AssertionError(f"unexpected request: {path}")


class CodeQLSuccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("check_codeql_success", SCRIPT)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)

    def assess(self, value=None):
        return self.verifier.assess(FakeApi(value or evidence()), PR, SHA)

    def test_complete_same_head_results_succeed(self):
        self.assertEqual(self.assess()["state"], "success")

    def test_non_success_or_missing_language_is_never_accepted(self):
        for conclusion in ("neutral", "skipped", "cancelled", "failure", None):
            with self.subTest(conclusion=conclusion):
                value = evidence()
                value["jobs"][1]["conclusion"] = conclusion
                self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["jobs"].pop()
        self.assertNotEqual(self.assess(value)["state"], "success")

    def test_running_or_newer_failed_run_cannot_reuse_older_success(self):
        for status, conclusion in (("in_progress", None), ("completed", "failure"), ("completed", "neutral")):
            with self.subTest(status=status, conclusion=conclusion):
                value = evidence()
                newer = copy.deepcopy(value["runs"][0])
                newer.update(
                    id=101,
                    created_at="2026-11-18T04:00:00Z",
                    updated_at="2026-11-18T04:05:00Z",
                    status=status,
                    conclusion=conclusion,
                )
                value["runs"].append(newer)
                self.assertNotEqual(self.assess(value)["state"], "success")

    def test_rerun_of_older_run_blocks_even_after_newer_run_succeeded(self):
        value = evidence()
        later = copy.deepcopy(value["runs"][0])
        later.update(
            id=101,
            created_at="2026-11-18T04:00:00Z",
            run_started_at="2026-11-18T04:00:00Z",
            updated_at="2026-11-18T04:05:00Z",
        )
        value["runs"].append(later)
        for analysis in value["analyses"]:
            analysis["created_at"] = "2026-11-18T04:01:00Z"
        value["aggregate"][0]["started_at"] = "2026-11-18T04:01:00Z"
        value["runs"][0].update(run_attempt=2, status="in_progress", conclusion=None)
        self.assertNotEqual(self.assess(value)["state"], "success")

    def test_analyses_must_be_current_exact_and_nonempty(self):
        mutations = (
            lambda a: a.update(commit_sha="b" * 40),
            lambda a: a.update(ref="refs/pull/99/head"),
            lambda a: a.update(category="/language:go"),
            lambda a: a.update(tool={"name": "Other"}),
            lambda a: a.update(analysis_key="untrusted:analyze"),
            lambda a: a.update(created_at="2026-11-18T02:59:59Z"),
            lambda a: a.update(rules_count=0),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = evidence()
                mutation(value["analyses"][1])
                self.assertNotEqual(self.assess(value)["state"], "success")

    def test_wrong_head_ref_workflow_or_attempt_fails_closed(self):
        for key, wrong in (("head_sha", "b" * 40), ("event", "push"), ("path", "malicious.yml")):
            with self.subTest(key=key):
                value = evidence()
                value["runs"][0][key] = wrong
                with self.assertRaises(ValueError):
                    self.assess(value)
        value = evidence()
        value["runs"][0]["head_branch"] = "refs/pull/7/head"
        self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["jobs"][0]["run_attempt"] = 2
        self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["jobs"][0]["run_id"] = 999
        self.assertNotEqual(self.assess(value)["state"], "success")

    def test_inconsistent_new_run_for_same_pr_does_not_fall_back_to_old_success(self):
        for key, wrong in (("head_sha", "b" * 40), ("workflow_id", 999), ("event", "push"), ("path", "other.yml")):
            with self.subTest(key=key):
                value = evidence()
                inconsistent = copy.deepcopy(value["runs"][0])
                inconsistent.update(id=101, created_at="2026-11-18T04:00:00Z", updated_at="2026-11-18T04:00:00Z")
                inconsistent[key] = wrong
                value["runs"].append(inconsistent)
                with self.assertRaises(ValueError):
                    self.assess(value)

    def test_head_or_run_changed_during_assessment_cannot_succeed(self):
        api = FakeApi(evidence())
        api.after_pr = copy.deepcopy(api.value["pr"])
        api.after_pr["head"]["sha"] = "b" * 40
        self.assertNotEqual(self.verifier.assess(api, PR, SHA)["state"], "success")
        api = FakeApi(evidence())
        api.after_runs = copy.deepcopy(api.value["runs"])
        api.after_runs[0].update(run_attempt=2, status="in_progress", conclusion=None)
        self.assertNotEqual(self.verifier.assess(api, PR, SHA)["state"], "success")

    def test_aggregate_must_succeed_for_exact_head_and_app(self):
        for mutation in (
            lambda check: check.update(conclusion="neutral"),
            lambda check: check.update(conclusion="skipped"),
            lambda check: check.update(status="in_progress"),
            lambda check: check.update(app={"id": 15368}),
            lambda check: check.update(head_sha="b" * 40),
            lambda check: check.update(pull_requests=[]),
            lambda check: check.update(started_at="2026-11-18T02:59:59Z"),
        ):
            with self.subTest(mutation=mutation):
                value = evidence()
                mutation(value["aggregate"][0])
                self.assertNotEqual(self.assess(value)["state"], "success")

    def test_unprocessed_sarif_or_open_codeql_finding_is_not_accepted(self):
        value = evidence()
        value["sarifs"][SARIFS[1]]["processing_status"] = "pending"
        self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["sarifs"][SARIFS[1]]["errors"] = ["failed"]
        self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["sarifs"][SARIFS[1]]["warnings"] = ["incomplete"]
        self.assertNotEqual(self.assess(value)["state"], "success")
        value = evidence()
        value["alerts"] = [{"number": 123, "state": "open", "tool": {"name": "CodeQL"}}]
        self.assertNotEqual(self.assess(value)["state"], "success")

    def test_bounded_wait_retries_pending_but_not_failure(self):
        elapsed = [0]

        def clock():
            return elapsed[0]

        def sleep(seconds):
            elapsed[0] += seconds

        responses = iter([{"state": "pending"}, {"state": "success"}])
        result = self.verifier.wait_until(lambda: next(responses), 10, interval=2, clock=clock, sleep=sleep)
        self.assertEqual(result["state"], "success")
        self.assertEqual(elapsed[0], 2)
        result = self.verifier.wait_until(lambda: {"state": "failure"}, 10, interval=2, clock=clock, sleep=sleep)
        self.assertEqual(result["state"], "failure")
        self.assertEqual(elapsed[0], 2)
        result = self.verifier.wait_until(lambda: {"state": "pending"}, 3, interval=2, clock=clock, sleep=sleep)
        self.assertEqual(result["state"], "failure")
        self.assertEqual(elapsed[0], 5)


if __name__ == "__main__":
    unittest.main()
