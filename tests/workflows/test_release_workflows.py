"""Execute release workflow shell with inert commands; never publish or use credentials."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
PUBLISH_STEP = "Mint short-lived crates.io token via Trusted Publishing and publish"


def step_script(filename: str, name: str) -> str:
    """Extract a known plain-name run step, rejecting unsupported YAML forms."""
    text = (WORKFLOWS / filename).read_text(encoding="utf-8")
    start = text.index(f"      - name: {name}\n")
    section = text[start:]
    end = re.search(r"\n(?:      - |  [a-zA-Z])", section)
    if end:
        section = section[: end.start()]
    if "continue-on-error:" in section:
        raise AssertionError("Release steps must not suppress failures")
    match = re.search(r"^        run: (.+)$", section, re.MULTILINE)
    if not match:
        raise AssertionError(f"Missing run command: {name}")
    if match.group(1) != "|":
        return match.group(1) + "\n"
    lines = section[match.end() + 1 :].splitlines()
    if any(line and not line.startswith("          ") for line in lines):
        raise AssertionError("Unexpected block scalar indentation")
    return textwrap.dedent("\n".join(lines)) + "\n"


STUB = r"""import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps([name, *args]) + "\n")
if name == "python":
    raise SystemExit(int(os.environ["GATE_EXIT"]) if "--release" in args else 0)
if name == "cargo":
    packages = [args[index + 1] for index, value in enumerate(args) if value == "-p"]
    if "--dry-run" in args and os.environ["FAIL_PACKAGE"] in packages:
        raise SystemExit(23)
    if "--dry-run" not in args and os.environ["FAIL_UPLOAD"] in packages:
        raise SystemExit(29)
if name == "curl":
    if "--data-binary" in args:
        sys.stdin.read()
    print('{"value":"synthetic-jwt","token":"synthetic-token"}')
if name == "jq":
    if "-nc" not in args:
        sys.stdin.read()
    print("synthetic-token" if ".token" in args else "synthetic-jwt")
"""


def execute(
    script: str,
    *,
    event: str = "workflow_dispatch",
    dry_run: str = "true",
    gate_exit: int = 0,
    fail_package: str = "",
    fail_upload: str = "",
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run actual shell logic against inert process-boundary stubs in a fresh tree."""
    script = script.replace("${{ github.event_name }}", event)
    script = script.replace("${{ inputs.dry_run }}", dry_run)
    if "${{" in script:
        raise AssertionError("Unexpected unrendered workflow expression")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("cargo", "curl", "jq", "sleep"):
            target = bin_dir / name
            target.write_text(f"#!{sys.executable} -S\n{STUB}", encoding="utf-8")
            target.chmod(0o700)
        interpreter = root / ".venv-release/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(f"#!{sys.executable} -S\n{STUB}", encoding="utf-8")
        interpreter.chmod(0o700)
        log = root / "commands.jsonl"
        env = {
            "PATH": str(bin_dir),
            "HOME": str(root),
            "LANG": "C.UTF-8",
            "COMMAND_LOG": str(log),
            "GITHUB_STEP_SUMMARY": str(root / "summary.md"),
            "GATE_EXIT": str(gate_exit),
            "FAIL_PACKAGE": fail_package,
            "FAIL_UPLOAD": fail_upload,
            "EVENT_NAME": event,
            "DRY_RUN": dry_run,
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "synthetic-request-token",
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://example.invalid/token?request=synthetic",
        }
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, commands


@unittest.skipUnless(os.name == "posix", "Hosted release workflows run on Ubuntu")
class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.publish = step_script("crates-publish.yml", PUBLISH_STEP)
        self.readiness = step_script("release-readiness.yml", "Check native migration policy")

    def test_dry_run_propagates_each_package_failure(self) -> None:
        for package in ("innovate-core", "innovate-tabular"):
            with self.subTest(package=package):
                result, commands = execute(self.publish, fail_package=package)
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertFalse(any(row[0] in ("curl", "jq") for row in commands))
                self.assertFalse(any(row[0] == "cargo" and "--dry-run" not in row for row in commands))

    def test_successful_dry_run_validates_both_without_credentials(self) -> None:
        result, commands = execute(self.publish)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            commands,
            [
                ["python", "scripts/check_migration.py", "--release"],
                ["cargo", "publish", "--locked", "--dry-run", "-p", "innovate-core", "-p", "innovate-tabular"],
            ],
        )

    def test_incomplete_migration_blocks_every_publishing_mode(self) -> None:
        for event, dry_run in (
            ("workflow_dispatch", "true"),
            ("workflow_dispatch", "false"),
            ("release", ""),
        ):
            with self.subTest(event=event, dry_run=dry_run):
                result, commands = execute(self.publish, event=event, dry_run=dry_run, gate_exit=17)
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(commands, [["python", "scripts/check_migration.py", "--release"]])

    def test_live_publish_preflights_both_before_token_request(self) -> None:
        for event, dry_run in (("workflow_dispatch", "false"), ("release", "")):
            with self.subTest(event=event):
                result, commands = execute(self.publish, event=event, dry_run=dry_run)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(commands[0], ["python", "scripts/check_migration.py", "--release"])
                self.assertEqual(
                    commands[1],
                    ["cargo", "publish", "--locked", "--dry-run", "-p", "innovate-core", "-p", "innovate-tabular"],
                )
                self.assertIn("::add-mask::synthetic-token", result.stdout)
                uploads = [row for row in commands if row[0] == "cargo" and "--dry-run" not in row]
                self.assertEqual(
                    uploads,
                    [
                        ["cargo", "publish", "--locked", "-p", "innovate-core", "-p", "innovate-tabular"],
                    ],
                )

    def test_live_mode_does_not_mint_token_when_packaging_fails(self) -> None:
        for package in ("innovate-core", "innovate-tabular"):
            with self.subTest(package=package):
                result, commands = execute(self.publish, event="release", dry_run="", fail_package=package)
                self.assertEqual(result.returncode, 23, result.stderr)
                self.assertFalse(any(row[0] in ("curl", "jq") for row in commands))

    def test_actual_upload_failure_is_not_swallowed(self) -> None:
        for package in ("innovate-core", "innovate-tabular"):
            with self.subTest(package=package):
                result, commands = execute(self.publish, dry_run="false", fail_upload=package)
                self.assertEqual(result.returncode, 29, result.stderr)
                self.assertEqual(
                    commands[-1],
                    ["cargo", "publish", "--locked", "-p", "innovate-core", "-p", "innovate-tabular"],
                )

    def test_invalid_publishing_mode_fails_without_side_effects(self) -> None:
        for event, dry_run in (
            ("pull_request", "true"),
            ("workflow_dispatch", ""),
            ("workflow_dispatch", "invalid"),
        ):
            with self.subTest(event=event, dry_run=dry_run):
                result, commands = execute(self.publish, event=event, dry_run=dry_run)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(commands, [])

    def test_readiness_remains_fail_closed_for_all_events(self) -> None:
        for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
            with self.subTest(event=event):
                result, commands = execute(self.readiness, event=event, gate_exit=17)
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                self.assertEqual(commands, [["python", "scripts/check_migration.py", "--release"]])

    def test_readiness_accepts_completed_migration(self) -> None:
        result, commands = execute(self.readiness)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [["python", "scripts/check_migration.py", "--release"]])

    def test_regressions_are_wired_into_required_policy_job(self) -> None:
        text = (WORKFLOWS / "security-quality.yml").read_text(encoding="utf-8")
        policy_job = text.split("  workflow-lint:\n", 1)[1].split("  python-quality:\n", 1)[0]
        self.assertIn("-m unittest discover -s tests/workflows -p 'test_release_workflows.py' -v", policy_job)
        publish = (WORKFLOWS / "crates-publish.yml").read_text(encoding="utf-8")
        self.assertIn("uv venv .venv-release --python 3.15.2", publish)
        self.assertIn("EVENT_NAME: ${{ github.event_name }}", publish)
        self.assertIn("DRY_RUN: ${{ inputs.dry_run }}", publish)
        self.assertNotIn("continue-on-error:", publish)


if __name__ == "__main__":
    unittest.main()
