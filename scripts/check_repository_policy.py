"""Check repository policy locally, optionally evaluating a supplied settings snapshot.

This command makes no network requests and never changes GitHub settings. A local
pass does not certify live enforcement, bot activity or coverage delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "edithatogo/innovate"
NO_SNAPSHOT = object()
CONTEXT_FILES = (
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
QUALIFIED_CHECKS = {"lint": 15368, "unit-tests-base (3.15.1)": 15368}
CLASSIC_CHECKS = {
    **QUALIFIED_CHECKS,
    "package": 15368,
    "CodeQL": 57789,
    "Secret Scan": 15368,
    "Workflow and policy lint": 15368,
    "Python quality harness": 15368,
}
BASE_RULES = {"deletion", "non_fast_forward", "pull_request", "required_status_checks"}
ADDITIONAL_LIVE_RULES = {"required_signatures", "required_linear_history", "required_conversation_resolution"}
OWNER_BYPASS = {"actor_type": "RepositoryRole", "actor_id": 5, "bypass_mode": "pull_request"}


def _object(value, path, findings, *, allowed=None, required=()):
    if not isinstance(value, dict):
        findings.append(f"{path}: expected an object")
        return False
    if allowed is not None:
        for key in value.keys() - allowed:
            findings.append(f"{path}: unknown field {key}")
    for key in required:
        if key not in value:
            findings.append(f"{path}: missing {key}")
    return True


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _read(root, name, findings, hashes):
    path = root / name
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("policy inputs must remain inside the checkout")
        content = path.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        text = content.decode("utf-8")
        if not text.strip():
            raise ValueError("file is empty")
        return text
    except (OSError, UnicodeError, ValueError) as error:
        findings.append(f"{name}: {error}")
        return None


def _local_json(root, name, findings, hashes):
    text = _read(root, name, findings, hashes)
    if text is None:
        return None
    try:
        return _json(text)
    except ValueError as error:
        findings.append(f"{name}: invalid JSON: {error}")
        return None


def _string_list(value, path, findings, *, nonempty=False):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        findings.append(f"{path}: expected a list of nonempty strings")
        return None
    if len(value) != len(set(value)) or (nonempty and not value):
        findings.append(f"{path}: requires unique{' nonempty' if nonempty else ''} entries")
    return value


def _review_parameters(value, path, findings, *, classic=False):
    owner_key = "require_code_owner_reviews" if classic else "require_code_owner_review"
    dismiss_key = "dismiss_stale_reviews" if classic else "dismiss_stale_reviews_on_push"
    required = {"required_approving_review_count", owner_key, "require_last_push_approval"}
    allowed = required | {
        dismiss_key,
        "required_review_thread_resolution",
        "allowed_merge_methods",
        "required_reviewers",
    }
    if classic:
        allowed |= {"url", "dismissal_restrictions", "bypass_pull_request_allowances"}
    if not _object(value, path, findings, allowed=allowed, required=required):
        return
    count = value.get("required_approving_review_count")
    if type(count) is not int or count != 0:
        findings.append(f"{path}.required_approving_review_count: must be integer zero for solo operation")
    for key in (owner_key, "require_last_push_approval"):
        if value.get(key) is not False:
            findings.append(f"{path}.{key}: mandatory human approvals must be false")
    for key in (dismiss_key, "required_review_thread_resolution"):
        if key in value and type(value[key]) is not bool:
            findings.append(f"{path}.{key}: expected a boolean")
    if "required_reviewers" in value and value["required_reviewers"] != []:
        findings.append(f"{path}.required_reviewers: no mandatory reviewer or team gate is permitted")
    if "allowed_merge_methods" in value:
        methods = _string_list(value["allowed_merge_methods"], f"{path}.allowed_merge_methods", findings, nonempty=True)
        if methods is not None and set(methods) - {"merge", "squash", "rebase"}:
            findings.append(f"{path}.allowed_merge_methods: unknown merge method")
    if classic:
        for key in ("dismissal_restrictions", "bypass_pull_request_allowances"):
            if key in value:
                _actors(value[key], f"{path}.{key}", findings)


def _status_checks(value, path, findings, *, local=False, classic=False):
    strict_key = "strict" if classic else "strict_required_status_checks_policy"
    list_key = "checks" if classic else "required_status_checks"
    id_key = "app_id" if classic else "integration_id"
    expected = CLASSIC_CHECKS if classic else QUALIFIED_CHECKS
    allowed = {strict_key, list_key} | (
        {"contexts", "contexts_url", "url"} if classic else {"do_not_enforce_on_create"}
    )
    if not _object(value, path, findings, allowed=allowed, required={strict_key, list_key}):
        return set()
    if value.get(strict_key) is not True:
        findings.append(f"{path}.{strict_key}: status checks must be strict")
    if "do_not_enforce_on_create" in value and value["do_not_enforce_on_create"] is not False:
        findings.append(f"{path}.do_not_enforce_on_create: checks must also apply on creation")
    checks = value.get(list_key)
    if not isinstance(checks, list) or not checks:
        findings.append(f"{path}.{list_key}: expected nonempty required checks")
        return set()
    names = set()
    qualified = set()
    for index, check in enumerate(checks):
        item_path = f"{path}.{list_key}[{index}]"
        if not _object(check, item_path, findings, allowed={"context", id_key}, required={"context", id_key}):
            continue
        name, application = check.get("context"), check.get(id_key)
        if not isinstance(name, str) or not name.strip():
            findings.append(f"{item_path}.context: expected a nonempty check name")
            continue
        if name in names:
            findings.append(f"{item_path}: duplicate required check {name}")
        names.add(name)
        if application is not None and (type(application) is not int or application <= 0):
            findings.append(f"{item_path}.{id_key}: expected a positive application id or null")
        if name in expected:
            if type(application) is int and application == expected[name]:
                qualified.add(name)
            else:
                findings.append(f"{item_path}.{id_key}: qualified check {name} must come from app {expected[name]}")
    if local and names != set(QUALIFIED_CHECKS):
        findings.append(
            f"{path}: proposed checks must be exactly {sorted(QUALIFIED_CHECKS)}; additional checks need qualification"
        )
    if classic:
        contexts = _string_list(value.get("contexts"), f"{path}.contexts", findings)
        if contexts is not None and set(contexts) != names:
            findings.append(f"{path}: contexts and application-bound checks disagree")
    return qualified


def _matches_main(value, path, findings):
    if value in {"~DEFAULT_BRANCH", "~ALL", "refs/heads/main"}:
        return True
    if value.startswith("refs/heads/") and not any(char in value for char in "*?[]\\"):
        return False
    findings.append(f"{path}: unsupported reference selector {value!r}; matching cannot be certified")
    return False


def _ruleset(value, path, findings, *, local=False):
    allowed = {"name", "target", "enforcement", "bypass_actors", "conditions", "rules"}
    if not local:
        allowed |= {
            "id",
            "node_id",
            "source",
            "source_type",
            "created_at",
            "updated_at",
            "_links",
            "current_user_can_bypass",
        }
    if not _object(
        value,
        path,
        findings,
        allowed=allowed,
        required={"name", "target", "enforcement", "conditions", "rules", "bypass_actors"},
    ):
        return set(), set()
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        findings.append(f"{path}.name: expected a nonempty ruleset name")
    if not isinstance(value.get("target"), str) or value["target"] not in {"branch", "tag", "push"}:
        findings.append(f"{path}.target: unknown ruleset target")
    if not isinstance(value.get("enforcement"), str) or value["enforcement"] not in {"active", "disabled", "evaluate"}:
        findings.append(f"{path}.enforcement: unknown ruleset enforcement")
    if value.get("target") != "branch" or value.get("enforcement") != "active":
        if local:
            findings.append(f"{path}: proposed ruleset must actively enforce branch rules")
        return set(), set()
    conditions = value.get("conditions")
    if not _object(conditions, f"{path}.conditions", findings, allowed={"ref_name"}, required={"ref_name"}):
        return set(), set()
    refs = conditions.get("ref_name")
    if not _object(
        refs, f"{path}.conditions.ref_name", findings, allowed={"include", "exclude"}, required={"include", "exclude"}
    ):
        return set(), set()
    includes = _string_list(refs.get("include"), f"{path}.conditions.ref_name.include", findings, nonempty=True)
    excludes = _string_list(refs.get("exclude"), f"{path}.conditions.ref_name.exclude", findings)
    if includes is None or excludes is None:
        return set(), set()
    included = [_matches_main(item, path, findings) for item in includes]
    excluded = [_matches_main(item, path, findings) for item in excludes]
    if local and (includes != ["~DEFAULT_BRANCH"] or excludes):
        findings.append(f"{path}: proposal must target only ~DEFAULT_BRANCH with no exclusions")
    if not any(included) or any(excluded):
        return set(), set()
    actors = value.get("bypass_actors")
    if (
        not isinstance(actors, list)
        or len(actors) != 1
        or not isinstance(actors[0], dict)
        or actors[0] != OWNER_BYPASS
        or type(actors[0].get("actor_id")) is not int
    ):
        findings.append(
            f"{path}.bypass_actors: retain only RepositoryRole 5 recovery in pull_request mode; no bot bypass"
        )
    rules = value.get("rules")
    if not isinstance(rules, list):
        findings.append(f"{path}.rules: expected a list of full rules")
        return set(), set()
    seen = set()
    checks = set()
    for index, rule in enumerate(rules):
        item_path = f"{path}.rules[{index}]"
        if not _object(rule, item_path, findings, allowed={"type", "parameters"}, required={"type"}):
            continue
        kind = rule.get("type")
        if not isinstance(kind, str) or kind not in BASE_RULES | (set() if local else ADDITIONAL_LIVE_RULES):
            findings.append(f"{item_path}: unknown or unqualified rule {kind!r}")
            continue
        if kind in seen:
            findings.append(f"{item_path}: duplicate rules of type {kind}")
        seen.add(kind)
        if kind == "pull_request":
            _review_parameters(rule.get("parameters"), f"{item_path}.pull_request", findings)
        elif kind == "required_status_checks":
            checks |= _status_checks(
                rule.get("parameters"), f"{item_path}.required_status_checks", findings, local=local
            )
        elif "parameters" in rule and rule["parameters"] != {}:
            findings.append(f"{item_path}: rule {kind} does not accept parameters")
    return seen, checks


def _actors(value, path, findings):
    allowed = {"users", "teams", "apps", "url", "users_url", "teams_url", "apps_url"}
    if not _object(value, path, findings, allowed=allowed, required={"users", "teams", "apps"}):
        return
    for kind in ("users", "teams", "apps"):
        actors = value.get(kind)
        if not isinstance(actors, list):
            findings.append(f"{path}.{kind}: expected a list")
        elif kind != "users" and actors:
            findings.append(f"{path}.{kind}: team gates and application exceptions require separate qualification")
        elif kind == "users" and any(
            not isinstance(actor, dict) or actor.get("login") != "edithatogo" for actor in actors
        ):
            findings.append(f"{path}.users: only the repository owner may retain a recovery exception")


def _protection(value, findings):
    path = "branch_protection"
    if value is None:
        findings.append(
            f"{path}: unavailable or unknown; a summary or local proposal cannot certify classic protection"
        )
        return
    optional_flags = {
        "enforce_admins",
        "required_linear_history",
        "block_creations",
        "required_conversation_resolution",
        "lock_branch",
        "allow_fork_syncing",
        "required_signatures",
    }
    required = {
        "allow_force_pushes",
        "allow_deletions",
        "required_status_checks",
        "required_pull_request_reviews",
        "restrictions",
    }
    if not _object(value, path, findings, allowed=required | optional_flags | {"url"}, required=required):
        return
    for key in {"allow_force_pushes", "allow_deletions"} | (optional_flags & value.keys()):
        flag = value.get(key)
        if not _object(flag, f"{path}.{key}", findings, allowed={"enabled", "url"}, required={"enabled"}):
            continue
        if type(flag.get("enabled")) is not bool:
            findings.append(f"{path}.{key}.enabled: expected a boolean")
        elif key in {"allow_force_pushes", "allow_deletions", "lock_branch"} and flag["enabled"]:
            findings.append(f"{path}.{key}: must be disabled for protected, operable main")
    checks = _status_checks(
        value.get("required_status_checks"), f"{path}.required_status_checks", findings, classic=True
    )
    if checks != set(CLASSIC_CHECKS):
        findings.append(
            f"{path}.required_status_checks: missing existing checks {sorted(set(CLASSIC_CHECKS) - checks)}"
        )
    reviews = value.get("required_pull_request_reviews")
    if reviews is not None:
        _review_parameters(reviews, f"{path}.required_pull_request_reviews", findings, classic=True)
    if value.get("restrictions") is not None:
        _actors(value["restrictions"], f"{path}.restrictions", findings)


def _snapshot(value, findings, now):
    required = {
        "schema_version",
        "repository",
        "default_branch",
        "observed_at",
        "commit_sha",
        "rulesets",
        "branch_protection",
        "errors",
    }
    informational = {
        "branch_protection_summary",
        "sources",
        "bot_inventory",
        "selected_ruleset_contexts",
        "excluded_new_required_contexts",
        "codecov",
        "settings_changes",
        "issue_completion",
        "limitations",
    }
    if not _object(value, "snapshot", findings, allowed=required | informational, required=required):
        return
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        findings.append("snapshot.schema_version: expected integer 1")
    if value.get("repository") != REPOSITORY:
        findings.append(f"snapshot.repository: expected {REPOSITORY}")
    if value.get("default_branch") != "main":
        findings.append("snapshot.default_branch: expected main")
    if not isinstance(value.get("commit_sha"), str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value["commit_sha"]):
        findings.append("snapshot.commit_sha: expected a full 40-digit hexadecimal commit")
    observed = value.get("observed_at")
    try:
        if not isinstance(observed, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", observed
        ):
            raise ValueError("expected an ISO UTC timestamp")
        timestamp = datetime.fromisoformat(observed)
        if now - timestamp > timedelta(days=7):
            raise ValueError("snapshot is older than seven days")
        if timestamp - now > timedelta(minutes=5):
            raise ValueError("snapshot is in the future")
    except ValueError as error:
        findings.append(f"snapshot.observed_at: {error}")
    if not isinstance(value.get("errors"), list) or value["errors"]:
        findings.append("snapshot.errors: collection errors prevent live verification")
    rulesets = value.get("rulesets")
    seen, checks = set(), set()
    if not isinstance(rulesets, list):
        findings.append("snapshot.rulesets: expected full ruleset objects")
    else:
        for index, ruleset in enumerate(rulesets):
            active_rules, active_checks = _ruleset(ruleset, f"rulesets[{index}]", findings)
            seen |= active_rules
            checks |= active_checks
    if not BASE_RULES <= seen:
        findings.append(f"snapshot.rulesets: active main rulesets lack {sorted(BASE_RULES - seen)}")
    if checks != set(QUALIFIED_CHECKS):
        findings.append("snapshot.rulesets: active main rulesets lack the qualified stable checks")
    _protection(value.get("branch_protection"), findings)


def build_report(root=ROOT, *, snapshot=NO_SNAPSHOT, now=None):
    """Evaluate checkout inputs and, when supplied, a sanitized settings snapshot."""
    root = Path(root)
    findings, hashes = [], {}
    for name in CONTEXT_FILES:
        _read(root, name, findings, hashes)
    renovate = _local_json(root, ".github/renovate.json", findings, hashes)
    if isinstance(renovate, dict):
        presets = _string_list(renovate.get("extends"), "Renovate.extends", findings, nonempty=True)
        if presets is not None and "github>edithatogo/renovate-config" not in presets:
            findings.append("Renovate.extends: missing github>edithatogo/renovate-config")
        if "enabled" in renovate and renovate["enabled"] is not True:
            findings.append("Renovate.enabled: the version-update owner must remain enabled")
    else:
        findings.append("Renovate: expected a configuration object")
    for name in (".github/dependabot.yml", ".github/dependabot.yaml"):
        if (root / name).exists() or (root / name).is_symlink():
            findings.append(f"Dependabot: competing version-update schedule remains at {name}")
    proposal = _local_json(root, ".github/rulesets/main.json", findings, hashes)
    seen, checks = _ruleset(proposal, ".github/rulesets/main.json", findings, local=True)
    if not BASE_RULES <= seen:
        findings.append(f".github/rulesets/main.json: missing rules {sorted(BASE_RULES - seen)}")
    if checks != set(QUALIFIED_CHECKS):
        findings.append(".github/rulesets/main.json: missing qualified stable checks")
    live = {
        "status": "not_checked",
        "source": "none",
        "findings": [],
        "limitation": "Local policy is not proof of live enforcement, bot activity or coverage delivery.",
    }
    if snapshot is not NO_SNAPSHOT:
        live["source"] = "supplied_snapshot"
        live["limitation"] = (
            "Evaluates supplied settings only; no live API request, bot or coverage-delivery check was performed."
        )
        _snapshot(snapshot, live["findings"], now or datetime.now(UTC))
        live["status"] = "fail" if live["findings"] else "pass"
        if isinstance(snapshot, dict):
            for key in ("observed_at", "commit_sha"):
                live[key] = snapshot.get(key)
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "scope": "local_policy" if snapshot is NO_SNAPSHOT else "local_and_supplied_snapshot",
        "status": "fail" if findings or live["status"] == "fail" else "pass",
        "local": {"status": "fail" if findings else "pass", "findings": findings},
        "live": live,
        "source_hashes": dict(sorted(hashes.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument("--snapshot", type=Path, help="evaluate a supplied sanitized GitHub settings snapshot")
    args = parser.parse_args(argv)
    value, load_error, snapshot_hash = NO_SNAPSHOT, None, None
    if args.snapshot is not None:
        try:
            raw = args.snapshot.read_bytes()
            snapshot_hash = hashlib.sha256(raw).hexdigest()
            value = _json(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            value, load_error = None, f"snapshot: cannot read valid JSON: {error}"
    report = build_report(snapshot=value)
    if load_error:
        report["live"]["findings"].insert(0, load_error)
    if snapshot_hash:
        report["live"]["snapshot_sha256"] = snapshot_hash
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Repository policy: {report['status'].upper()}")
        print(f"Local files: {report['local']['status'].upper()}")
        print(f"Live settings: {report['live']['status'].upper()} ({report['live']['source']})")
        print(report["live"]["limitation"])
        for scope in ("local", "live"):
            for finding in report[scope]["findings"]:
                print(f"- {scope}: {finding}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
