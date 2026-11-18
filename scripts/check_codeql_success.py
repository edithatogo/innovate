"""Read-only, fail-closed check of CodeQL evidence for an exact pull-request head.

This script does not create a status check or change branch protection. Its
result is a point-in-time observation, not continuous enforcement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "edithatogo/innovate"
WORKFLOW = "dynamic/github-code-scanning/codeql"
LANGUAGES = ("python", "rust")
PAGE_SIZE = 100
MAX_PAGES = 10


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    """Minimal fixed-origin client; untrusted redirects and oversized lists fail closed."""

    def __init__(self, token):
        if not token:
            raise ValueError("GITHUB_TOKEN is required")
        self._token = token
        self._opener = build_opener(_NoRedirect())

    def get(self, path, params=None):
        if not path.startswith("/") or path.startswith("//") or "?" in path:
            raise ValueError("invalid API path")
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"https://api.github.com/repos/{REPOSITORY}{path}{query}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "innovate-codeql-evidence/1",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with self._opener.open(request, timeout=30) as response:
            return json.load(response)

    def items(self, path, key=None, params=None):
        result = []
        for page in range(1, MAX_PAGES + 1):
            value = self.get(path, {**(params or {}), "per_page": PAGE_SIZE, "page": page})
            batch = value if key is None else value.get(key) if isinstance(value, dict) else None
            if not isinstance(batch, list):
                raise ValueError("malformed paginated API response")
            result.extend(batch)
            if len(batch) < PAGE_SIZE:
                return result
        raise ValueError("API result exceeds the supported pagination bound")


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("missing GitHub timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("GitHub timestamp has no timezone")
    return result


def _matching_pr(value, sha):
    return (
        isinstance(value, dict)
        and value.get("state") == "open"
        and isinstance(value.get("head"), dict)
        and value["head"].get("sha") == sha
        and isinstance(value.get("base"), dict)
        and value["base"].get("ref") == "main"
    )


def _latest_run(runs, workflow_id, ref, sha):
    candidates = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("malformed CodeQL run")
        if run.get("head_branch") != ref:
            continue
        if (
            run.get("workflow_id") != workflow_id
            or run.get("path") != WORKFLOW
            or run.get("event") != "dynamic"
            or run.get("head_sha") != sha
        ):
            raise ValueError("inconsistent CodeQL run identity")
        candidates.append(run)
    for run in candidates:
        if type(run.get("id")) is not int or type(run.get("run_attempt")) is not int:
            raise ValueError("invalid CodeQL run identity")
        _timestamp(run.get("created_at"))
        _timestamp(run.get("updated_at"))
    if not candidates:
        return None
    # A rerun of an older workflow keeps its original creation time. Never let a
    # later successful run hide another still-running attempt on this PR head.
    incomplete = [run for run in candidates if run.get("status") != "completed"]
    return max(incomplete or candidates, key=lambda run: (_timestamp(run["updated_at"]), run["id"]))


def _aggregate(api, pr_number, sha):
    checks = api.items(f"/commits/{sha}/check-runs", "check_runs", {"check_name": "CodeQL", "filter": "latest"})
    matched = []
    for check in checks:
        if (
            not isinstance(check, dict)
            or check.get("name") != "CodeQL"
            or check.get("head_sha") != sha
            or not isinstance(check.get("app"), dict)
            or check["app"].get("id") != 57789
            or not isinstance(check.get("pull_requests"), list)
        ):
            continue
        if any(
            isinstance(pr, dict)
            and pr.get("number") == pr_number
            and isinstance(pr.get("head"), dict)
            and pr["head"].get("sha") == sha
            and isinstance(pr.get("base"), dict)
            and pr["base"].get("ref") == "main"
            for pr in check["pull_requests"]
        ):
            matched.append(check)
    if not matched or any(check.get("status") != "completed" for check in matched):
        return None
    for check in matched:
        if type(check.get("id")) is not int:
            raise ValueError("invalid aggregate check identity")
        _timestamp(check.get("started_at"))
    return max(matched, key=lambda check: (_timestamp(check["started_at"]), check["id"]))


def assess(api, pr_number, sha):
    """Observe the latest CodeQL run and its uploaded evidence for one PR head."""
    if (
        type(pr_number) is not int
        or pr_number <= 0
        or not isinstance(sha, str)
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
    ):
        raise ValueError("expected a positive PR number and a lowercase 40-character SHA")
    ref = f"refs/pull/{pr_number}/head"
    if not _matching_pr(api.get(f"/pulls/{pr_number}"), sha):
        return {"state": "failure", "reason": "pull request is closed, retargeted or on a different head"}

    workflows = api.items("/actions/workflows", "workflows")
    candidates = [
        workflow
        for workflow in workflows
        if isinstance(workflow, dict) and workflow.get("path") == WORKFLOW and workflow.get("state") == "active"
    ]
    if len(candidates) != 1 or type(candidates[0].get("id")) is not int:
        return {"state": "failure", "reason": "default CodeQL workflow is missing or ambiguous"}
    workflow_id = candidates[0]["id"]
    run_path = f"/actions/workflows/{workflow_id}/runs"
    run_params = {"head_sha": sha}
    latest = _latest_run(api.items(run_path, "workflow_runs", run_params), workflow_id, ref, sha)
    if latest is None:
        return {"state": "pending", "reason": "no matching CodeQL run"}
    if latest.get("status") != "completed":
        return {"state": "pending", "reason": "latest CodeQL run has not completed"}
    if latest.get("conclusion") != "success":
        return {"state": "failure", "reason": "latest CodeQL run did not succeed"}

    run_id, attempt = latest["id"], latest["run_attempt"]
    started = _timestamp(latest.get("run_started_at"))
    jobs = api.items(f"/actions/runs/{run_id}/jobs", "jobs", {"filter": "latest"})
    for language in LANGUAGES:
        matched = [job for job in jobs if isinstance(job, dict) and job.get("name") == f"Analyze ({language})"]
        if (
            len(matched) != 1
            or matched[0].get("status") != "completed"
            or matched[0].get("conclusion") != "success"
            or matched[0].get("run_attempt") != attempt
            or matched[0].get("run_id") != run_id
        ):
            return {"state": "failure", "reason": f"{language} analysis job did not succeed on the latest attempt"}

    analyses = api.items("/code-scanning/analyses", params={"ref": ref, "tool_name": "CodeQL"})
    for language in LANGUAGES:
        matched = [
            analysis
            for analysis in analyses
            if isinstance(analysis, dict)
            and analysis.get("ref") == ref
            and analysis.get("commit_sha") == sha
            and analysis.get("category") == f"/language:{language}"
            and analysis.get("analysis_key") == f"{WORKFLOW}:analyze"
            and isinstance(analysis.get("tool"), dict)
            and analysis["tool"].get("name") == "CodeQL"
            and type(analysis.get("rules_count")) is int
            and analysis["rules_count"] > 0
            and _timestamp(analysis.get("created_at")) >= started
        ]
        if not matched:
            return {"state": "pending", "reason": f"current {language} analysis is not available"}
        analysis = max(matched, key=lambda value: _timestamp(value["created_at"]))
        sarif = analysis.get("sarif_id")
        if not isinstance(sarif, str) or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", sarif
        ):
            return {"state": "failure", "reason": f"{language} analysis has no valid SARIF identity"}
        processing = api.get(f"/code-scanning/sarifs/{sarif}")
        if not isinstance(processing, dict) or processing.get("processing_status") != "complete":
            return {"state": "pending", "reason": f"{language} SARIF has not finished processing"}
        if processing.get("errors") not in (None, []) or processing.get("warnings") not in (None, []):
            return {"state": "failure", "reason": f"{language} SARIF has processing diagnostics"}

    aggregate = _aggregate(api, pr_number, sha)
    if aggregate is None or _timestamp(aggregate.get("started_at")) < started:
        return {"state": "pending", "reason": "current CodeQL aggregate is not available"}
    if aggregate.get("conclusion") != "success":
        return {"state": "failure", "reason": "CodeQL aggregate did not succeed"}
    alerts = api.items("/code-scanning/alerts", params={"ref": ref, "tool_name": "CodeQL", "state": "open"})
    if alerts:
        return {"state": "failure", "reason": "open CodeQL findings exist on the pull-request head"}

    current_pr = api.get(f"/pulls/{pr_number}")
    current = _latest_run(api.items(run_path, "workflow_runs", run_params), workflow_id, ref, sha)
    current_aggregate = _aggregate(api, pr_number, sha)
    if (
        not _matching_pr(current_pr, sha)
        or current is None
        or current.get("id") != run_id
        or current.get("run_attempt") != attempt
        or current.get("status") != "completed"
        or current.get("conclusion") != "success"
        or current_aggregate is None
        or current_aggregate.get("id") != aggregate["id"]
        or current_aggregate.get("conclusion") != "success"
    ):
        return {"state": "pending", "reason": "pull request or latest run changed during assessment"}
    return {"state": "success", "head_sha": sha, "run_id": run_id, "run_attempt": attempt}


def wait_until(observe, timeout, *, interval=30, clock=time.monotonic, sleep=time.sleep):
    """Wait only for pending evidence; failures and API errors are never retried."""
    deadline = clock() + timeout
    while True:
        result = observe()
        if result.get("state") != "pending":
            return result
        remaining = deadline - clock()
        if remaining <= 0:
            return {"state": "failure", "reason": "timed out waiting for CodeQL evidence"}
        sleep(min(interval, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 2400:
        parser.error("--wait-seconds must be between 0 and 2400")
    try:
        api = GitHub(os.environ.get("GITHUB_TOKEN"))
        if args.wait_seconds:
            result = wait_until(lambda: assess(api, args.pr, args.head), args.wait_seconds)
        else:
            result = assess(api, args.pr, args.head)
    except HTTPError as error:
        print(json.dumps({"state": "error", "reason": f"GitHub API HTTP {error.code}"}))
        return 2
    except OSError, URLError, ValueError, TypeError, KeyError, json.JSONDecodeError:
        print(json.dumps({"state": "error", "reason": "unable to verify GitHub evidence"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
