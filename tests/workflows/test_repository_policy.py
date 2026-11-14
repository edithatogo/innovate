"""Behavioral checks for local policy and supplied GitHub-setting evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_repository_policy.py"
NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)
CONTEXT = (
    "AGENTS.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "conductor/index.md",
    "conductor/product.md",
    "conductor/product-guidelines.md",
    "conductor/tech-stack.md",
    "conductor/workflow.md",
    "conductor/tracks.md",
    "conductor/execution-contract-20261107.md",
)


def ruleset():
    return {
        "name": "Default branch safety",
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [{"actor_type": "RepositoryRole", "actor_id": 5, "bypass_mode": "pull_request"}],
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                    "required_review_thread_resolution": True,
                },
            },
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": True,
                    "required_status_checks": [
                        {"context": "lint", "integration_id": 15368},
                        {"context": "unit-tests-base (3.15.1)", "integration_id": 15368},
                    ],
                },
            },
        ],
    }


def snapshot():
    return {
        "schema_version": 1,
        "repository": "edithatogo/innovate",
        "default_branch": "main",
        "observed_at": NOW.isoformat(),
        "commit_sha": "a" * 40,
        "rulesets": [ruleset()],
        "branch_protection": {
            "required_status_checks": {
                "strict": True,
                "contexts": [
                    "lint",
                    "unit-tests-base (3.15.1)",
                    "CodeQL",
                    "package",
                    "Secret Scan",
                    "Workflow and policy lint",
                    "Python quality harness",
                ],
                "checks": [
                    {"context": "lint", "app_id": 15368},
                    {"context": "unit-tests-base (3.15.1)", "app_id": 15368},
                    {"context": "CodeQL", "app_id": 57789},
                    {"context": "package", "app_id": 15368},
                    {"context": "Secret Scan", "app_id": 15368},
                    {"context": "Workflow and policy lint", "app_id": 15368},
                    {"context": "Python quality harness", "app_id": 15368},
                ],
            },
            "required_pull_request_reviews": {
                "required_approving_review_count": 0,
                "require_code_owner_reviews": False,
                "require_last_push_approval": False,
                "dismiss_stale_reviews": False,
                "dismissal_restrictions": {"users": [], "teams": [], "apps": []},
                "bypass_pull_request_allowances": {"users": [], "teams": [], "apps": []},
            },
            "enforce_admins": {"enabled": False},
            "allow_force_pushes": {"enabled": False},
            "allow_deletions": {"enabled": False},
            "restrictions": None,
        },
        "errors": [],
    }


class RepositoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "The repository-policy checker has not been implemented")
        spec = importlib.util.spec_from_file_location("repository_policy", SCRIPT)
        self.policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.policy)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in CONTEXT:
            self.write(name, "# Canonical context\n")
        self.write_json(".github/renovate.json", {"extends": ["github>edithatogo/renovate-config"]})
        self.write_json(".github/rulesets/main.json", ruleset())

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_json(self, name, value):
        self.write(name, json.dumps(value))

    def report(self, value=None, *, remote=False):
        if remote:
            return self.policy.build_report(self.root, snapshot=value, now=NOW)
        return self.policy.build_report(self.root, now=NOW)

    def assert_failed(self, report, scope, text):
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report[scope]["status"], "fail", report)
        self.assertIn(text, "\n".join(report[scope]["findings"]), report)

    def test_local_success_does_not_certify_live_settings(self):
        report = self.report()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["local"]["status"], "pass")
        self.assertEqual(report["live"]["status"], "not_checked")
        self.assertEqual(report["scope"], "local_policy")
        self.assertEqual(len(report["source_hashes"]["AGENTS.md"]), 64)
        before = report["source_hashes"]["AGENTS.md"]
        self.write("AGENTS.md", "# Changed canonical instructions\n")
        self.assertNotEqual(self.report()["source_hashes"]["AGENTS.md"], before)

    def test_context_files_must_exist_and_have_content(self):
        for content in (None, " \n"):
            with self.subTest(content=content):
                if content is None:
                    (self.root / "AGENTS.md").unlink()
                else:
                    self.write("AGENTS.md", content)
                self.assert_failed(self.report(), "local", "AGENTS.md")

    def test_renovate_requires_central_inheritance(self):
        for value in (
            {"extends": []},
            {"extends": "github>edithatogo/renovate-config"},
            {"extends": ["github>edithatogo/renovate-config"], "enabled": False},
        ):
            with self.subTest(value=value):
                self.write_json(".github/renovate.json", value)
                self.assert_failed(self.report(), "local", "Renovate")

    def test_competing_dependabot_schedules_fail_for_both_extensions(self):
        for name in (".github/dependabot.yml", ".github/dependabot.yaml"):
            with self.subTest(name=name):
                self.write(name, "version: 2\nupdates: []\n")
                self.assert_failed(self.report(), "local", "Dependabot")
                (self.root / name).unlink()

    def test_local_rules_reject_loss_duplicates_and_unknown_rules(self):
        mutations = {
            "missing deletion": lambda value: value["rules"].pop(0),
            "duplicate deletion": lambda value: value["rules"].append({"type": "deletion"}),
            "unknown rule": lambda value: value["rules"].append({"type": "unknown_policy"}),
            "malformed rule": lambda value: value["rules"].append("deletion"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = ruleset()
                mutate(value)
                self.write_json(".github/rulesets/main.json", value)
                self.assert_failed(self.report(), "local", "rules")

    def test_local_checks_are_exact_qualified_and_strict(self):
        mutations = {
            "missing": lambda value: value["required_status_checks"].pop(),
            "extra": lambda value: value["required_status_checks"].append(
                {"context": "unqualified", "integration_id": 15368}
            ),
            "wrong app": lambda value: value["required_status_checks"][0].update(integration_id=123),
            "noninteger app": lambda value: value["required_status_checks"][0].update(integration_id="15368"),
            "duplicate": lambda value: value["required_status_checks"].append(
                copy.deepcopy(value["required_status_checks"][0])
            ),
            "not strict": lambda value: value.update(strict_required_status_checks_policy=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = ruleset()
                mutate(value["rules"][3]["parameters"])
                self.write_json(".github/rulesets/main.json", value)
                self.assert_failed(self.report(), "local", "check")

    def test_local_review_and_bypass_policy_fails_closed(self):
        for field, bad in (
            ("required_approving_review_count", 1),
            ("required_approving_review_count", False),
            ("require_code_owner_review", True),
            ("require_last_push_approval", True),
            ("required_reviewers", [{"team": "maintainers"}]),
            ("unknown_option", False),
        ):
            with self.subTest(field=field, bad=bad):
                value = ruleset()
                value["rules"][2]["parameters"][field] = bad
                self.write_json(".github/rulesets/main.json", value)
                self.assert_failed(self.report(), "local", "pull_request")
        for actors in (
            [],
            [{"actor_type": "Integration", "actor_id": 15368, "bypass_mode": "always"}],
            [{"actor_type": "RepositoryRole", "actor_id": 5, "bypass_mode": "always"}],
        ):
            with self.subTest(actors=actors):
                value = ruleset()
                value["bypass_actors"] = actors
                self.write_json(".github/rulesets/main.json", value)
                self.assert_failed(self.report(), "local", "bypass")

    def test_valid_supplied_snapshot_accepts_additional_automated_controls(self):
        value = snapshot()
        value["rulesets"][0]["rules"].append({"type": "required_signatures"})
        value["rulesets"][0]["rules"][3]["parameters"]["required_status_checks"].append(
            {"context": "Secret Scan", "integration_id": 15368}
        )
        report = self.report(value, remote=True)
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["live"]["status"], "pass")
        self.assertEqual(report["scope"], "local_and_supplied_snapshot")
        self.assertEqual(report["live"]["source"], "supplied_snapshot")

    def test_classic_protection_preserves_all_preexisting_checks_and_their_apps(self):
        for removed in ("package", "Secret Scan", "Workflow and policy lint", "Python quality harness", "CodeQL"):
            with self.subTest(removed=removed):
                value = snapshot()
                checks = value["branch_protection"]["required_status_checks"]
                checks["contexts"].remove(removed)
                checks["checks"] = [check for check in checks["checks"] if check["context"] != removed]
                self.assert_failed(self.report(value, remote=True), "live", "required_status_checks")
        value = snapshot()
        value["branch_protection"]["required_status_checks"]["checks"][2]["app_id"] = 15368
        self.assert_failed(self.report(value, remote=True), "live", "CodeQL")

    def test_receipt_information_is_not_used_as_live_protection(self):
        value = snapshot()
        value.update(
            sources=[],
            bot_inventory=[],
            selected_ruleset_contexts=[],
            excluded_new_required_contexts=[],
            codecov={},
            settings_changes=[],
            issue_completion={},
            limitations=[],
        )
        self.assertEqual(self.report(value, remote=True)["status"], "pass")
        value["branch_protection"] = None
        value["settings_changes"] = {"claims_all_settings_pass": True}
        self.assert_failed(self.report(value, remote=True), "live", "branch_protection")

    def test_missing_disabled_or_excluded_live_rules_cannot_pass(self):
        mutations = {
            "missing": lambda value: value.update(rulesets=[]),
            "disabled": lambda value: value["rulesets"][0].update(enforcement="disabled"),
            "evaluate": lambda value: value["rulesets"][0].update(enforcement="evaluate"),
            "wrong target": lambda value: value["rulesets"][0].update(target="tag"),
            "excluded": lambda value: value["rulesets"][0]["conditions"]["ref_name"].update(
                exclude=["refs/heads/main"]
            ),
            "other branch": lambda value: value["rulesets"][0]["conditions"]["ref_name"].update(
                include=["refs/heads/develop"]
            ),
            "missing force rule": lambda value: value["rulesets"][0]["rules"].pop(1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = snapshot()
                mutate(value)
                self.assert_failed(self.report(value, remote=True), "live", "ruleset")

    def test_other_live_rulesets_cannot_add_human_approval_or_bot_bypass(self):
        for mutation in ("approval", "bot"):
            with self.subTest(mutation=mutation):
                value = snapshot()
                second = ruleset()
                if mutation == "approval":
                    second["rules"][2]["parameters"]["required_approving_review_count"] = 1
                else:
                    second["bypass_actors"].append(
                        {"actor_type": "Integration", "actor_id": 99, "bypass_mode": "always"}
                    )
                value["rulesets"].append(second)
                self.assert_failed(self.report(value, remote=True), "live", "rulesets[1]")

    def test_unavailable_protection_and_collection_errors_remain_unknown(self):
        value = snapshot()
        value["branch_protection"] = None
        value["branch_protection_summary"] = {"required_status_checks": {"contexts": ["lint"]}}
        self.assert_failed(self.report(value, remote=True), "live", "branch_protection")
        value = snapshot()
        value["errors"] = [{"status": 403, "resource": "branch protection"}]
        self.assert_failed(self.report(value, remote=True), "live", "errors")

    def test_snapshot_identity_schema_and_commit_are_validated(self):
        for key, bad in (
            ("repository", "someone/else"),
            ("default_branch", "master"),
            ("schema_version", True),
            ("schema_version", 2),
            ("commit_sha", "main"),
            ("rulesets", {}),
            ("errors", None),
        ):
            with self.subTest(key=key):
                value = snapshot()
                value[key] = bad
                self.assert_failed(self.report(value, remote=True), "live", key)
        self.assert_failed(self.report(None, remote=True), "live", "snapshot")

    def test_snapshot_timestamp_is_recent_utc_and_not_in_future(self):
        for bad in (
            (NOW - timedelta(days=8)).isoformat(),
            (NOW + timedelta(hours=1)).isoformat(),
            "2026-11-14T12:00:00",
            "2026-11-14T22:00:00+10:00",
            "not a time",
        ):
            with self.subTest(bad=bad):
                value = snapshot()
                value["observed_at"] = bad
                self.assert_failed(self.report(value, remote=True), "live", "observed_at")

    def test_classic_protection_cannot_add_human_gates_or_allow_destructive_pushes(self):
        for key in ("allow_force_pushes", "allow_deletions"):
            with self.subTest(key=key):
                value = snapshot()
                value["branch_protection"][key]["enabled"] = True
                self.assert_failed(self.report(value, remote=True), "live", key)
        for key, bad in (
            ("required_approving_review_count", 1),
            ("require_code_owner_reviews", True),
            ("require_last_push_approval", True),
        ):
            with self.subTest(key=key):
                value = snapshot()
                value["branch_protection"]["required_pull_request_reviews"][key] = bad
                self.assert_failed(self.report(value, remote=True), "live", key)
        value = snapshot()
        value["branch_protection"]["restrictions"] = {"teams": [{"slug": "owners"}], "users": [], "apps": []}
        self.assert_failed(self.report(value, remote=True), "live", "restrictions")

    def test_unknown_live_rule_or_malformed_protection_cannot_pass(self):
        value = snapshot()
        value["rulesets"][0]["rules"].append({"type": "unknown_policy"})
        self.assert_failed(self.report(value, remote=True), "live", "unknown_policy")
        for bad in ({}, [], "unavailable"):
            with self.subTest(bad=bad):
                value = snapshot()
                value["branch_protection"] = bad
                self.assert_failed(self.report(value, remote=True), "live", "branch_protection")

    def test_malformed_or_duplicate_json_keys_fail_without_traceback(self):
        self.write(".github/rulesets/main.json", '{"enforcement":"disabled","enforcement":"active"}')
        self.assert_failed(self.report(), "local", "duplicate")
        self.write(".github/rulesets/main.json", "{broken")
        self.assert_failed(self.report(), "local", "JSON")

    def test_cli_json_and_snapshot_errors_have_truthful_exit_codes(self):
        self.write("scripts/check_repository_policy.py", SCRIPT.read_text(encoding="utf-8"))
        command = [sys.executable, str(self.root / "scripts/check_repository_policy.py"), "--json"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["live"]["status"], "not_checked")
        self.assertEqual(completed.stderr, "")
        for name, text in (("missing.json", None), ("broken.json", "{"), ("null.json", "null")):
            with self.subTest(name=name):
                if text is not None:
                    self.write(name, text)
                completed = subprocess.run(
                    command + ["--snapshot", str(self.root / name)], capture_output=True, text=True, check=False
                )
                self.assertEqual(completed.returncode, 1, completed)
                self.assertEqual(json.loads(completed.stdout)["live"]["status"], "fail")
                self.assertNotIn("Traceback", completed.stderr)
        for field in ("target", "enforcement"):
            with self.subTest(field=field):
                malformed = ruleset()
                malformed[field] = []
                self.write_json(".github/rulesets/main.json", malformed)
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 1, completed)
                self.assertEqual(json.loads(completed.stdout)["local"]["status"], "fail")
                self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
