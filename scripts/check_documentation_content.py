"""Audit documentation inventory, tutorial structure and executable source bindings.

An audit inventories debt; strict mode rejects all current debt. Ratchet mode
rejects new or changed debt against a reviewed source-bound baseline and never
labels remaining debt complete. Structural checks do not establish pedagogical
quality, empirical validity or execution. Only --execute runs Python examples;
declared Julia/R execution also requires complete, source-bound native receipts.
"""

from __future__ import annotations

import argparse
import ast
import doctest
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE = Path("docs/astro-site/src/content/docs")
CONTRACT = Path("docs/contracts/content-quality.json")
BASELINE = Path("docs/contracts/content-baseline.json")
API_CONTRACT = Path("docs/contracts/api-documentation.json")
POLYGLOT_INVENTORY = Path("docs/contracts/polyglot-api-documentation.json")
POLYGLOT_CHECKER = Path("scripts/check_polyglot_api_documentation.py")
LANGUAGE_RUNNERS = {
    "julia": "scripts/check_julia_documentation.jl",
    "r": "scripts/check_r_documentation.R",
}
GENERATED_API = {
    "generator": "scripts/generate_api_reference.py",
    "manifest": "docs/astro-site/src/content/docs/api/generated/manifest.json",
}
SECTIONS = ("learning outcomes", "prerequisites", "steps", "expected results", "interpretation", "limitations")
KINDS = {"tutorial", "reference", "index", "historical"}
LEGACY_NOTICE = ":::caution[Pre-0.7 reference]"
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME)\b|\bcoming soon\b", re.IGNORECASE)
EXECUTION_RUNNER = r"""
import ast
import json
import pathlib
import runpy
import sys

source = pathlib.Path(sys.argv[1]).resolve()
receipt = pathlib.Path(sys.argv[2])
lines = {
    item.lineno for item in ast.walk(ast.parse(source.read_text()))
    if isinstance(item, ast.Assert) and any(
        isinstance(child, (ast.Name, ast.Call)) for child in ast.walk(item.test)
    )
}
executed = set()

def trace(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == str(source) and frame.f_lineno in lines:
        executed.add(frame.f_lineno)
    return trace

sys.argv = [str(source)]
sys.settrace(trace)
try:
    runpy.run_path(str(source), run_name="__main__")
finally:
    sys.settrace(None)
receipt.write_text(json.dumps({"executed_assertion_lines": sorted(executed)}))
"""
DOCTEST_RUNNER = r"""
import doctest
import hashlib
import importlib
import json
import pathlib
import sys

if sys.version_info[:3] != (3, 15, 2):
    raise RuntimeError("Executable documentation requires Python 3.15.2.")
module_name, expected_sha256, receipt_filename = sys.argv[1:]
module = importlib.import_module(module_name)
installed_path = pathlib.Path(module.__file__).resolve()
actual_sha256 = hashlib.sha256(installed_path.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise RuntimeError("Installed binding source differs from the documentation source.")
tests = doctest.DocTestFinder().find(module)
expected_attempts = sum(len(test.examples) for test in tests)
if expected_attempts == 0:
    raise RuntimeError("No executable docstring examples were discovered.")
if any(example.options.get(doctest.SKIP) for test in tests for example in test.examples):
    raise RuntimeError("Skipped docstring examples do not qualify as executed documentation.")
result = doctest.testmod(module, verbose=False)
receipt = {
    "module": module_name,
    "installed_source": str(installed_path),
    "installed_source_sha256": actual_sha256,
    "python_version": sys.version.split()[0],
    "attempted": result.attempted,
    "expected_attempts": expected_attempts,
    "failed": result.failed,
}
pathlib.Path(receipt_filename).write_text(json.dumps(receipt))
if result.failed or result.attempted != expected_attempts:
    raise SystemExit(1)
"""


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contract_digest(contract: dict[str, Any]) -> str:
    return _digest(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())


def _normalized_digest(path: Path) -> str:
    """Match the native runners' UTF-8, CRLF/CR-to-LF text hashing contract."""
    return _digest(path.read_text(encoding="utf-8").encode("utf-8"))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)


def _required_languages(root: Path) -> list[str]:
    policy = _read_json(root / API_CONTRACT)
    languages = policy.get("generated_reference_languages") if isinstance(policy, dict) else None
    if (
        not isinstance(languages, list)
        or not languages
        or not all(isinstance(language, str) for language in languages)
        or len(set(languages)) != len(languages)
        or "python" not in languages
        or not set(languages) <= {"python", *LANGUAGE_RUNNERS}
    ):
        raise ValueError("generated_reference_languages must uniquely declare python and any required julia/r scope.")
    return languages


def _safe_relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def _validate_contract(contract: object) -> list[str]:
    if (
        not isinstance(contract, dict)
        or not {"schema_version", "pages", "examples"} <= set(contract)
        or not set(contract) <= {"schema_version", "pages", "examples", "generated_api"}
    ):
        return ["Contract requires schema_version, pages and examples; only generated_api is optional."]
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        return ["Unsupported schema_version; expected integer 1."]
    if not isinstance(contract["pages"], dict) or not isinstance(contract["examples"], dict):
        return ["pages and examples must be objects keyed by repository paths."]
    errors = []
    if "generated_api" in contract and contract["generated_api"] != GENERATED_API:
        errors.append("generated_api must name the fixed, verified API generator and output manifest.")
    for path, entry in contract["pages"].items():
        if not _safe_relative(path) or Path(path).suffix not in {".md", ".mdx"} or Path(path).parts[0] == "latest":
            errors.append(f"Invalid current page path: {path!r}.")
        if (
            not isinstance(entry, dict)
            or not set(entry) <= {"kind", "example", "examples", "reason"}
            or not isinstance(entry.get("kind"), str)
            or entry["kind"] not in KINDS
        ):
            errors.append(f"Invalid page classification: {path!r}.")
            continue
        if entry["kind"] == "historical" and not isinstance(entry.get("reason"), str):
            errors.append(f"Historical page needs a reason: {path!r}.")
        if "example" in entry and (
            not _safe_relative(entry["example"]) or entry["example"] not in contract["examples"]
        ):
            errors.append(f"Page example is not registered: {path!r}.")
        if "examples" in entry:
            additional = entry["examples"]
            if not isinstance(additional, list) or not additional:
                errors.append(f"Page examples must be a nonempty list of registered sources: {path!r}.")
            elif any(not _safe_relative(name) or name not in contract["examples"] for name in additional):
                errors.append(f"Page examples contain an unregistered source: {path!r}.")
            elif len(additional) != len(set(additional)) or entry.get("example") in additional:
                errors.append(f"Page example declarations must be unique, including the primary example: {path!r}.")
    for path, entry in contract["examples"].items():
        if not _safe_relative(path) or Path(path).parts[0] != "examples" or Path(path).suffix != ".py":
            errors.append(f"Invalid example path: {path!r}.")
        if not isinstance(entry, dict) or set(entry) != {
            "status",
            "language",
            "network",
            "timeout_seconds",
            "expected_stdout",
        }:
            errors.append(f"Invalid example metadata: {path!r}.")
            continue
        if entry["status"] != "runnable" or entry["language"] != "python" or entry["network"] is not False:
            errors.append(f"Only explicitly runnable offline Python examples are supported: {path!r}.")
        if type(entry["timeout_seconds"]) is not int or not 1 <= entry["timeout_seconds"] <= 120:
            errors.append(f"Example timeout must be an integer in [1, 120]: {path!r}.")
        if not isinstance(entry["expected_stdout"], str) or not 1 <= len(entry["expected_stdout"]) <= 65536:
            errors.append(f"Example needs a bounded explicit expected_stdout string: {path!r}.")
    return errors


def _verified_generated(root: Path) -> tuple[set[str], list[str]]:
    """Validate recomputed API artifacts before recognizing any generated page."""
    generator = root / GENERATED_API["generator"]
    manifest_path = root / GENERATED_API["manifest"]
    if not generator.is_file() or not manifest_path.is_file():
        return set(), ["API generator or generated manifest is absent; build references before this gate."]
    if not generator.resolve().is_relative_to(root) or not manifest_path.resolve().is_relative_to(root):
        return set(), ["Generator and manifest must remain within the repository."]
    try:
        spec = importlib.util.spec_from_file_location("_innovate_generated_api_contract", generator)
        if spec is None or spec.loader is None:
            raise ValueError("API generator cannot be loaded.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        errors = module.check_generated(root)
        if errors:
            return set(), list(errors)
        artifacts = module.build_artifacts(root)
        expected = {path for path in artifacts if Path(path).suffix in {".md", ".mdx"}}
        manifest = json.loads(manifest_path.read_text())
        declared = {(manifest_path.parent.relative_to(root) / path).as_posix() for path in manifest["pages"]}
        if not expected or expected != declared:
            raise ValueError("Generated manifest and recomputed page inventory differ.")
        for path in expected:
            if not _safe_relative(path) or not (root / path).resolve().is_relative_to(root):
                raise ValueError("Generated page path escapes the repository.")
        return expected, []
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, SyntaxError) as exc:
        return set(), [f"Generated API validation failed: {exc}"]


def _visible_text(text: str) -> str:
    """Remove MDX/HTML comments while preserving literal fenced-code content."""
    visible: list[str] = []
    offset = 0
    fence_start = re.compile(r"[ \t]{0,3}(`{3,}|~{3,})[^\n]*", re.MULTILINE)
    comment_start = re.compile(r"<!--|\{\s*/\*")
    while offset < len(text):
        if offset == 0 or text[offset - 1] == "\n":
            fence = fence_start.match(text, offset)
            if fence:
                marker = fence.group(1)
                closing = re.search(
                    rf"^[ \t]{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$",
                    text[fence.end() :],
                    re.MULTILINE,
                )
                end = fence.end() + closing.end() if closing else len(text)
                visible.append(text[offset:end])
                offset = end
                continue
        if text[offset] in {"<", "{"}:
            comment = comment_start.match(text, offset)
            if comment:
                closing_pattern = r"--!?>" if comment.group() == "<!--" else r"\*/\s*\}"
                closing = re.search(closing_pattern, text[comment.end() :])
                end = comment.end() + closing.end() if closing else len(text)
                visible.append("\n" * text[offset:end].count("\n"))
                offset = end
                continue
        visible.append(text[offset])
        offset += 1
    return "".join(visible)


def _fences(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate fenced blocks so headings in examples cannot satisfy prose rules."""
    prose: list[str] = []
    code: list[str] = []
    blocks = []
    marker = ""
    language = ""
    for line in _visible_text(text).splitlines():
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if not marker and fence:
            marker = fence.group(1)
            language = fence.group(2).strip().split(" ", 1)[0]
            code = []
        elif (
            marker
            and fence
            and fence.group(1)[0] == marker[0]
            and len(fence.group(1)) >= len(marker)
            and not fence.group(2).strip()
        ):
            blocks.append((language, "\n".join(code).strip()))
            marker = ""
        elif marker:
            code.append(line)
        else:
            prose.append(line)
    return "\n".join(prose), blocks


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, re.DOTALL)
    if match is None:
        return {}
    return {
        key: value.strip().strip("\"'")
        for key, value in re.findall(r"^(title|description):\s*(.+)$", match.group(1), re.MULTILINE)
    }


def _has_source_binding(page: Path, text: str, example: Path) -> bool:
    # Support a complete verbatim fence or a Starlight Code component backed by
    # Vite's raw import. A prose mention of a source filename is not evidence.
    prose, blocks = _fences(text)
    if any(lang in {"python", "py"} and code == example.read_text().strip() for lang, code in blocks):
        return True
    if not re.search(
        r"^import\s*\{[^}]*\bCode\b[^}]*\}\s*from\s*['\"]@astrojs/starlight/components['\"]", prose, re.MULTILINE
    ):
        return False
    for identifier, relative in re.findall(
        r"^import\s+([A-Za-z_$][\w$]*)\s+from\s*['\"]([^'\"]+)\?raw['\"]", prose, re.MULTILINE
    ):
        if (page.parent / relative).resolve() == example.resolve() and re.search(
            rf"<Code\s[^>]*\bcode\s*=\s*\{{\s*{re.escape(identifier)}\s*\}}", prose
        ):
            return True
    return False


def _example_assertions(text: str) -> bool:
    parsed = ast.parse(text)
    return any(
        isinstance(node, ast.Assert) and any(isinstance(child, (ast.Name, ast.Call)) for child in ast.walk(node.test))
        for node in ast.walk(parsed)
    )


def build_report(root: Path, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Discover every source page/example and return reproducible structural debt."""
    root = root.resolve()
    if contract is None:
        try:
            contract = json.loads((root / CONTRACT).read_text())
        except (OSError, ValueError) as exc:
            contract = {}
            read_error = str(exc)
        else:
            read_error = ""
    else:
        read_error = ""
    violations: list[dict[str, str]] = []

    def add(path: str, rule: str, detail: str) -> None:
        candidate = root / path
        digest = (
            _digest(candidate.read_bytes())
            if candidate.is_file() and candidate.resolve().is_relative_to(root)
            else "missing"
        )
        violations.append({"path": path, "rule": rule, "detail": detail, "source_sha256": digest})

    errors = _validate_contract(contract)
    if read_error:
        errors.append(read_error)
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract_sha256": _contract_digest(contract),
        "complete": False,
        "current_structure_complete": False,
        "full_scope_structure_complete": False,
        "current_python_content_complete": False,
        "current_content_complete": False,
        "full_scope_complete": False,
        "execution_status": "not_run",
        "python_execution_status": "not_run",
        "example_execution_status": "not_run",
        "api_doctest_status": "not_run",
        "required_languages": [],
        "language_receipts": {},
        "quality_claim": "Structural evidence only; expert review and executable outcomes remain separate.",
        "counts": {},
        "pages": [],
        "examples": [],
        "violations": violations,
    }
    if errors:
        add(str(CONTRACT), "invalid-contract", " ".join(errors))
        return report
    try:
        report["required_languages"] = _required_languages(root)
        report["language_contract_sha256"] = _normalized_digest(root / API_CONTRACT)
        report["language_receipts"] = {
            language: {"status": "not_run", "qualified": False}
            for language in report["required_languages"]
            if language != "python"
        }
    except (OSError, ValueError) as exc:
        add(str(API_CONTRACT), "invalid-language-contract", str(exc))
    generated: set[str] = set()
    if "generated_api" in contract:
        generated, generated_errors = _verified_generated(root)
        report["generated_api_status"] = "verified" if not generated_errors else "not_built_or_invalid"
        if generated_errors:
            add(GENERATED_API["manifest"], "generated-api-evidence", " ".join(generated_errors))
    else:
        report["generated_api_status"] = "not_configured"
    pages = sorted((root / SITE).rglob("*"))
    pages = [path for path in pages if path.is_file() and path.suffix in {".md", ".mdx"}]
    examples = sorted(path for path in (root / "examples").rglob("*") if path.is_file() and path.name != ".gitkeep")
    if not pages:
        add(str(SITE), "empty-page-inventory", "No documentation source pages were discovered.")
    if not examples:
        add("examples", "empty-example-inventory", "No example sources were discovered.")
    discovered_pages = {path.relative_to(root / SITE).as_posix() for path in pages}
    discovered_examples = {path.relative_to(root).as_posix() for path in examples}
    unbound_python_blocks = 0
    for missing in set(contract["pages"]) - discovered_pages:
        add((SITE / missing).as_posix(), "missing-page", "Manifest page does not exist.")
    for missing in set(contract["examples"]) - discovered_examples:
        add(missing, "missing-example", "Manifest example does not exist.")
    for path in pages:
        relative = path.relative_to(root / SITE).as_posix()
        repo_path = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to(root):
            add(repo_path, "source-outside-repository", "Symlink resolves outside repository.")
            continue
        text = path.read_text()
        if relative.startswith("latest/"):
            report["pages"].append({"path": repo_path, "classification": "historical_version"})
            continue
        entry = contract["pages"].get(relative)
        classification = "generated_api" if repo_path in generated else entry["kind"] if entry else "unclassified"
        report["pages"].append(
            {"path": repo_path, "classification": classification, "source_sha256": _digest(path.read_bytes())}
        )
        if entry is None and classification != "generated_api":
            add(repo_path, "unclassified-page", "New current-path page requires an explicit reviewed classification.")
            continue
        if classification == "generated_api":
            entry = {"kind": "reference"}
        example_name = entry.get("example")
        example_names = ([example_name] if example_name is not None else []) + entry.get("examples", [])
        source_paths = {
            name: root / name
            for name in example_names
            if (root / name).is_file() and (root / name).resolve().is_relative_to(root)
        }
        missing_bindings = [
            name
            for name in example_names
            if name not in source_paths or not _has_source_binding(path, text, source_paths[name])
        ]
        if missing_bindings:
            add(
                repo_path,
                "example-source-binding",
                "Each declared example must be imported or displayed completely; missing bindings: "
                + ", ".join(missing_bindings)
                + ".",
            )
        if classification == "historical":
            if LEGACY_NOTICE not in _fences(text)[0]:
                add(
                    repo_path,
                    "historical-notice",
                    "Historical current-path page must visibly identify its pre-0.7 scope.",
                )
            continue
        metadata = _frontmatter(text)
        for field in ("title", "description"):
            value = metadata.get(field, "")
            if len(value.strip()) < 3 or value.strip() in {"|", ">"} or PLACEHOLDER.search(value):
                add(repo_path, f"frontmatter-{field}", f"A substantive scalar {field} is required.")
        prose, blocks = _fences(text)
        if PLACEHOLDER.search(prose):
            add(repo_path, "placeholder-content", "Unresolved placeholder in current documentation prose.")
        is_tutorial = classification == "tutorial"
        if relative.startswith("tutorials/") and Path(relative).stem != "index" and not is_tutorial:
            add(
                repo_path,
                "tutorial-classification",
                "A current tutorial cannot be reclassified as reference to evade tutorial checks.",
            )
        if is_tutorial:
            headings = list(re.finditer(r"^##\s+(.+?)\s*$", prose, re.MULTILINE))
            bodies = {
                heading.group(1).casefold(): prose[
                    heading.end() : headings[index + 1].start() if index + 1 < len(headings) else len(prose)
                ]
                for index, heading in enumerate(headings)
            }
            for section in SECTIONS:
                body = bodies.get(section, "")
                if len(re.findall(r"\b\w+\b", body)) < 5 or PLACEHOLDER.search(body):
                    add(
                        repo_path,
                        "section-" + section.replace(" ", "-"),
                        f"Tutorial needs substantive '{section}' guidance.",
                    )
        if is_tutorial and example_name is None:
            add(repo_path, "tutorial-example", "Tutorial requires a registered executable example.")
        python_blocks = [code for lang, code in blocks if lang in {"python", "py"}]
        if python_blocks and classification != "generated_api":
            source_texts = {source.read_text().strip() for source in source_paths.values()}
            unbound_count = sum(code not in source_texts for code in python_blocks)
            unbound_python_blocks += unbound_count
            if unbound_count:
                add(
                    repo_path,
                    "unbound-python-example",
                    f"{unbound_count} displayed Python blocks lack complete registered-source bindings; "
                    "a raw import elsewhere does not cover unrelated blocks.",
                )
        for code in python_blocks:
            try:
                parsed = ast.parse(code)
            except SyntaxError:
                add(repo_path, "python-example-syntax", "A current Python fenced example is not syntactically valid.")
                break
            imports = set()
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imports.update(item.name.split(".")[0] for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            retired = imports & {"pandas", "xarray", "jax", "numpyro", "blackjax", "mesa", "ndlib"}
            if retired:
                add(
                    repo_path,
                    "retired-runtime-example",
                    "Current example imports retired runtime dependencies: " + ", ".join(sorted(retired)),
                )
                break
    for path in examples:
        relative = path.relative_to(root).as_posix()
        entry = contract["examples"].get(relative)
        report["examples"].append({"path": relative, "classification": "runnable" if entry else "unclassified"})
        if not path.resolve().is_relative_to(root):
            add(relative, "source-outside-repository", "Symlink resolves outside repository.")
        elif entry is None:
            add(
                relative,
                "unclassified-example",
                "Every file in examples requires an explicit supported execution contract.",
            )
        else:
            try:
                has_assertion = _example_assertions(path.read_text())
            except SyntaxError, UnicodeError:
                add(relative, "example-syntax", "Example is not valid Python source.")
            else:
                if not has_assertion:
                    add(
                        relative,
                        "example-assertion",
                        "Runnable example must check a nonconstant condition, not merely print a result.",
                    )
    report["violations"] = sorted(violations, key=lambda item: (item["path"], item["rule"], item["detail"]))
    report["current_structure_complete"] = not violations
    report["counts"] = {
        "pages": len(report["pages"]),
        "current_pages": sum(
            item["classification"] not in {"historical", "historical_version"} for item in report["pages"]
        ),
        "historical_pages": sum(item["classification"] == "historical" for item in report["pages"]),
        "versioned_archive_pages": sum(item["classification"] == "historical_version" for item in report["pages"]),
        "examples": len(examples),
        "generated_api_pages": sum(item["classification"] == "generated_api" for item in report["pages"]),
        "unbound_python_blocks": unbound_python_blocks,
        "violations": len(violations),
    }
    report["full_scope_structure_complete"] = not violations and not (
        report["counts"]["historical_pages"] or report["counts"]["versioned_archive_pages"]
    )
    report["historical_scope"] = (
        "Historical pages are inventoried and visibly distinguished; their scientific accuracy, "
        "migration and executable qualification remain unverified."
    )
    return report


def make_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """Prepare debt for explicit review; this helper is never called by CI."""
    return {"schema_version": 1, "contract_sha256": report["contract_sha256"], "violations": report["violations"]}


def evaluate_ratchet(report: dict[str, Any], baseline: object) -> dict[str, Any]:
    """Compare exact source-bound debt without turning tolerated debt into completion."""
    if (
        not isinstance(baseline, dict)
        or set(baseline) != {"schema_version", "contract_sha256", "violations"}
        or type(baseline["schema_version"]) is not int
        or baseline["schema_version"] != 1
        or not isinstance(baseline["violations"], list)
    ):
        return {
            "passed": False,
            "reason": "Malformed baseline; review a complete debt inventory.",
            "new_or_changed": [],
        }
    if baseline["contract_sha256"] != report["contract_sha256"]:
        return {
            "passed": False,
            "reason": "Content policy changed; explicit baseline review is required.",
            "new_or_changed": [],
        }
    expected_keys = {"path", "rule", "detail", "source_sha256"}
    for item in baseline["violations"]:
        if (
            not isinstance(item, dict)
            or set(item) != expected_keys
            or not all(isinstance(value, str) for value in item.values())
        ):
            return {"passed": False, "reason": "Malformed baseline violation.", "new_or_changed": []}
    identities = [(item["path"], item["rule"]) for item in baseline["violations"]]
    if len(set(identities)) != len(identities):
        return {"passed": False, "reason": "Duplicate baseline exceptions must be removed.", "new_or_changed": []}
    unknown = [item for item in report["violations"] if item not in baseline["violations"]]
    stale = [item for item in baseline["violations"] if item not in report["violations"]]
    # Invalid inventories and policy are never grandfathered into a baseline.
    hard_rules = {
        "invalid-contract",
        "invalid-language-contract",
        "unclassified-page",
        "unclassified-example",
        "missing-page",
        "missing-example",
        "empty-page-inventory",
        "empty-example-inventory",
        "source-outside-repository",
        "generated-api-evidence",
    }
    hard = [item for item in report["violations"] if item["rule"] in hard_rules]
    return {
        "passed": not unknown and not hard and not stale,
        "reason": "No new structural debt; completeness is reported separately."
        if not unknown and not hard and not stale
        else "New, changed or unclassifiable debt, or stale exceptions requiring removal.",
        "new_or_changed": unknown or hard,
        "stale": stale,
    }


def execute_examples(root: Path, contract: dict[str, Any], python: str) -> list[dict[str, Any]]:
    """Execute registered examples with a chosen installed interpreter and timeout.

    Temporary working directories and isolated Python avoid repository shadowing.
    The network:false declaration is a review contract, not an OS network sandbox.
    """
    if _validate_contract(contract):
        return [{"status": "not_run", "reason": "Invalid execution contract."}]
    results = []
    for relative, entry in sorted(contract["examples"].items()):
        source = (root / relative).resolve()
        result: dict[str, Any] = {
            "path": relative,
            "status": "not_run",
            "source_sha256": _digest(source.read_bytes())
            if source.is_file() and source.is_relative_to(root.resolve())
            else "missing",
        }
        if not source.is_file() or not source.is_relative_to(root.resolve()):
            result["reason"] = "Missing or external source."
            results.append(result)
            continue
        try:
            if not _example_assertions(source.read_text()):
                raise ValueError("Example has no nonconstant assertion.")
        except (ValueError, SyntaxError, UnicodeError) as exc:
            result["reason"] = str(exc)
            results.append(result)
            continue
        env = {
            key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"}
        }
        env.update({"CI": "true", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"})
        try:
            with tempfile.TemporaryDirectory(prefix="innovate-doc-example-") as workdir:
                receipt_path = Path(workdir) / "assertion-receipt.json"
                process = subprocess.run(
                    [python, "-I", "-c", EXECUTION_RUNNER, str(source), str(receipt_path)],
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=entry["timeout_seconds"],
                    check=False,
                )
                receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            result.update(
                {
                    "status": "passed" if process.returncode == 0 else "failed",
                    "returncode": process.returncode,
                    "stdout": process.stdout[-65536:],
                    "stderr": process.stderr[-65536:],
                    "python": python,
                }
            )
            result["executed_assertion_lines"] = receipt.get("executed_assertion_lines", [])
            result["stdout_matches_contract"] = process.stdout == entry["expected_stdout"]
            if not result["stdout_matches_contract"]:
                result.update(
                    {"status": "failed", "reason": "Example stdout differs from the documented output contract."}
                )
            if process.returncode == 0 and not result["executed_assertion_lines"]:
                result.update({"status": "failed", "reason": "No outcome assertion was executed."})
            if not source.is_file() or _digest(source.read_bytes()) != result["source_sha256"]:
                result.update({"status": "failed", "reason": "Example source changed during execution."})
        except subprocess.TimeoutExpired:
            result.update({"status": "failed", "reason": "Example exceeded its declared timeout."})
        except (OSError, ValueError) as exc:
            result["reason"] = str(exc)
        results.append(result)
    if "generated_api" in contract:
        results.extend(execute_api_doctests(root, python))
    return results


def execute_api_doctests(root: Path, python: str) -> list[dict[str, Any]]:
    """Execute every generated source's doctests against identical installed files."""
    generated, errors = _verified_generated(root.resolve())
    if errors or not generated:
        return [{"kind": "api_doctests", "status": "not_run", "reason": "Generated API evidence is invalid."}]
    manifest = json.loads((root / GENERATED_API["manifest"]).read_text())
    results: list[dict[str, Any]] = []
    for relative, expected_sha in sorted(manifest["sources"].items()):
        if Path(relative).suffix != ".py":
            continue
        source = root / relative
        tree = ast.parse(source.read_text())
        docs = [
            ast.get_docstring(node) or ""
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not any(doctest.DocTestParser().get_examples(doc) for doc in docs):
            continue
        package_path = Path(relative).relative_to("bindings/python")
        parts = package_path.with_suffix("").parts
        module_name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        result: dict[str, Any] = {
            "kind": "api_doctests",
            "path": relative,
            "status": "not_run",
            "source_sha256": expected_sha,
        }
        env = {
            key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"}
        }
        env.update({"CI": "true", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"})
        try:
            with tempfile.TemporaryDirectory(prefix="innovate-api-doctest-") as directory:
                receipt_path = Path(directory) / "doctest-receipt.json"
                process = subprocess.run(
                    [python, "-I", "-c", DOCTEST_RUNNER, module_name, expected_sha, str(receipt_path)],
                    cwd=directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            result.update(receipt)
            passed = (
                process.returncode == 0
                and receipt.get("failed") == 0
                and receipt.get("attempted", 0) > 0
                and receipt.get("attempted") == receipt.get("expected_attempts")
                and receipt.get("installed_source_sha256") == expected_sha
            )
            result.update(
                {
                    "status": "passed" if passed else "failed",
                    "returncode": process.returncode,
                    "stdout": process.stdout[-65536:],
                    "stderr": process.stderr[-65536:],
                }
            )
            if not source.is_file() or _digest(source.read_bytes()) != expected_sha:
                result.update({"status": "failed", "reason": "Binding source changed during execution."})
        except subprocess.TimeoutExpired:
            result.update({"status": "failed", "reason": "API doctests exceeded their declared timeout."})
        except (OSError, ValueError) as exc:
            result["reason"] = str(exc)
        results.append(result)
    if not results:
        return [{"kind": "api_doctests", "status": "not_run", "reason": "No generated API doctest examples found."}]
    return results


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _verified_native_inventory(root: Path) -> tuple[dict[str, Any], str]:
    """Recompute the fixed structural inventory and independently rehash its inputs."""
    path = root / POLYGLOT_CHECKER
    _require(
        path.is_file() and path.resolve().is_relative_to(root), "Polyglot inventory checker is absent or external."
    )
    spec = importlib.util.spec_from_file_location("_innovate_native_docs_inventory", path)
    _require(spec is not None and spec.loader is not None, "Polyglot inventory checker cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    errors = module.check_inventory(root)
    _require(isinstance(errors, list) and not errors, f"Polyglot inventory is not current: {errors}")
    inventory_path = root / POLYGLOT_INVENTORY
    _require(inventory_path.resolve().is_relative_to(root), "Polyglot inventory must remain within the repository.")
    inventory = _read_json(inventory_path)
    _require(isinstance(inventory, dict), "Polyglot inventory must be an object.")
    _require(
        type(inventory.get("schema_version")) is int and inventory["schema_version"] == 1,
        "Unsupported polyglot inventory schema.",
    )
    sources = inventory.get("sources")
    _require(isinstance(sources, dict) and bool(sources), "Polyglot source hash inventory is empty or invalid.")
    _require(
        {str(POLYGLOT_CHECKER), *LANGUAGE_RUNNERS.values()} <= set(sources),
        "Polyglot inventory omits a checker or native runner.",
    )
    for relative, expected in sources.items():
        _require(_safe_relative(relative) and _valid_hash(expected), "Invalid polyglot source path or hash.")
        source = root / relative
        _require(
            source.is_file() and source.resolve().is_relative_to(root),
            f"Polyglot source is absent or external: {relative}.",
        )
        _require(_normalized_digest(source) == expected, f"Polyglot source changed: {relative}.")
    return inventory, _normalized_digest(inventory_path)


def _verify_native_receipt(
    root: Path, language: str, receipt: Any, inventory: dict[str, Any], inventory_sha: str
) -> None:
    """Check coherent execution evidence; receipts are not cryptographically authenticated."""
    _require(isinstance(receipt, dict), "Receipt must be a JSON object.")
    _require(
        type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1, "Unsupported receipt schema."
    )
    _require(receipt.get("language") == language, "Receipt language differs from its declared scope.")
    _require(
        receipt.get("status") == "passed" and receipt.get("qualified") is True,
        "Native runner did not qualify this language.",
    )
    _require(receipt.get("runner") == LANGUAGE_RUNNERS[language], "Unexpected native documentation runner.")
    _require(receipt.get("inventory_sha256") == inventory_sha, "Receipt inventory hash is stale or absent.")
    _require(
        receipt.get("source_sha256") == inventory["sources"],
        "Receipt source hashes differ from the complete current inventory.",
    )
    for field in (
        "runtime",
        "platform",
        "package_version",
        "json_version" if language == "julia" else "jsonlite_version",
    ):
        _require(
            isinstance(receipt.get(field), str) and bool(receipt[field].strip()), f"Receipt lacks {field} identity."
        )
    _require(re.search(r"\d+\.\d+", receipt["runtime"]) is not None, "Receipt runtime has no version identity.")
    native = receipt.get("native_library")
    _require(
        isinstance(native, dict) and isinstance(native.get("path"), str) and _valid_hash(native.get("sha256")),
        "Receipt lacks native binary identity.",
    )
    native_name = Path(native["path"].replace("\\", "/")).name
    _require(
        native_name in {"libinnovate_ffi.so", "libinnovate_ffi.dylib", "innovate_ffi.dll", "libinnovate_ffi.dll"},
        "Receipt names an unexpected native library.",
    )
    staged = root / "target/release" / native_name
    _require(
        staged.is_file() and staged.resolve().is_relative_to(root),
        "Current staged release native library is absent or external.",
    )
    _require(
        _digest(staged.read_bytes()) == native["sha256"],
        "Receipt native binary differs from the current staged release.",
    )
    names = [item["name"] for item in inventory["symbols"][language]]
    observed_names = receipt.get("exported_names_checked")
    _require(bool(names) and len(names) == len(set(names)), "Native exported-symbol inventory is empty or duplicated.")
    _require(
        isinstance(observed_names, list) and all(isinstance(name, str) for name in observed_names),
        "Receipt export checks are malformed.",
    )
    _require(
        len(observed_names) == len(names) and set(observed_names) == set(names),
        "Receipt does not cover every current export exactly once.",
    )
    if language == "julia":
        _require(
            type(receipt.get("docstrings_checked")) is int and receipt["docstrings_checked"] == len(names),
            "Receipt omits Julia docstring checks.",
        )
    else:
        manuals = {path for path in inventory["sources"] if path.startswith("bindings/r/man/") and path.endswith(".Rd")}
        observed_manuals = receipt.get("manuals_checked")
        _require(
            isinstance(observed_manuals, list) and all(isinstance(path, str) for path in observed_manuals),
            "Receipt manual checks are malformed.",
        )
        _require(
            bool(manuals) and len(observed_manuals) == len(manuals) and set(observed_manuals) == manuals,
            "Receipt does not cover every current R manual exactly once.",
        )
        installed = receipt.get("installed_code_sha256")
        _require(
            isinstance(installed, dict)
            and len(installed) == 2
            and all(_valid_hash(value) for value in installed.values()),
            "Receipt lacks installed R code identity.",
        )
        _require(
            {Path(path).suffix for path in installed} == {".rdb", ".rdx"},
            "Receipt installed R code identities are incomplete.",
        )
    expected = inventory["native_examples"][language]
    observed = receipt.get("examples")
    _require(isinstance(expected, list) and bool(expected), "Native example inventory is empty or invalid.")
    _require(
        isinstance(observed, list) and len(observed) == len(expected),
        "Receipt has missing or additional native examples.",
    )
    expected_by_id = {item["id"]: item for item in expected}
    _require(len(expected_by_id) == len(expected), "Native inventory example IDs are duplicated.")
    seen: set[str] = set()
    assertion_total = 0
    for item in observed:
        _require(isinstance(item, dict) and isinstance(item.get("id"), str), "Receipt example is malformed.")
        identity = item["id"]
        _require(
            identity in expected_by_id and identity not in seen, "Receipt has unknown or duplicate native example IDs."
        )
        seen.add(identity)
        example = expected_by_id[identity]
        _require(
            isinstance(example["code"], str) and _digest(example["code"].encode("utf-8")) == example["code_sha256"],
            "Inventory example code hash is invalid.",
        )
        _require(
            example["source_sha256"] == inventory["sources"][example["source"]],
            "Inventory example source hash is invalid.",
        )
        _require(item.get("status") == "passed", f"Native example did not pass: {identity}.")
        _require(
            all(item.get(key) == example[key] for key in ("source", "source_line", "code_sha256"))
            and _positive_int(item.get("source_line")),
            f"Native example source binding differs: {identity}.",
        )
        count = example["assertion_count"]
        _require(
            _positive_int(count)
            and _positive_int(item.get("assertions_executed"))
            and item["assertions_executed"] == count,
            f"Native example assertions are missing, skipped or repeated: {identity}.",
        )
        assertion_total += count
    for field, total in (("examples_executed", len(expected)), ("assertions_executed", assertion_total)):
        _require(
            type(receipt.get(field)) is int and receipt[field] == total,
            f"Receipt {field} total differs from the exact inventory.",
        )
    counts = inventory["counts"]
    _require(
        counts[f"{language}_example_blocks"] == len(expected)
        and counts[f"{language}_example_assertions"] == assertion_total,
        "Native inventory example totals are inconsistent.",
    )


def verify_language_receipts(root: Path, directory: Path | None, required: list[str]) -> dict[str, dict[str, Any]]:
    """Validate complete Julia/R runner evidence against current sources and binaries.

    This aggregates trusted CI artifacts. Matching hashes establish consistency,
    not artifact authenticity or the scientific validity of asserted outcomes.
    """
    root = root.resolve()
    results = {
        language: {"status": "not_run", "qualified": False, "reason": "Native documentation receipt was not supplied."}
        for language in required
        if language != "python"
    }
    if not results or directory is None:
        return results
    try:
        inventory, inventory_sha = _verified_native_inventory(root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, SyntaxError) as exc:
        return {language: {"status": "failed", "qualified": False, "reason": str(exc)} for language in results}
    for language in results:
        path = directory / f"{language}-documentation.json"
        if not path.is_file():
            results[language]["reason"] = f"Required native documentation receipt is missing: {path}."
            continue
        try:
            receipt = _read_json(path)
            _verify_native_receipt(root, language, receipt, inventory, inventory_sha)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            results[language] = {"status": "failed", "qualified": False, "reason": str(exc), "path": str(path)}
        else:
            results[language] = {"status": "passed", "qualified": True, "path": str(path), "receipt": receipt}
    return results


def main(argv: list[str] | None = None) -> int:
    """Run a machine-readable audit, strict gate or reviewed-debt ratchet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "strict", "ratchet"), default="strict")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--language-receipt-dir",
        type=Path,
        help="Julia/R runner receipt directory; required by --execute when those languages are declared.",
    )
    args = parser.parse_args(argv)
    report = build_report(args.root)
    passed = (
        report["complete"]
        if args.mode == "strict"
        else not any(item["rule"] in {"invalid-contract", "invalid-language-contract"} for item in report["violations"])
    )
    if args.mode == "ratchet":
        try:
            baseline = json.loads((args.baseline or args.root / BASELINE).read_text())
        except OSError, ValueError:
            baseline = None
        report["ratchet"] = evaluate_ratchet(report, baseline)
        passed = report["ratchet"]["passed"]
    if args.execute:
        try:
            contract = json.loads((args.root / CONTRACT).read_text())
        except OSError, ValueError:
            contract = {}
        # Keep a virtual-environment executable's symlink path. Resolving it
        # launches the base interpreter and loses the environment's packages.
        report["executions"] = execute_examples(
            args.root, contract, str(Path(args.python).absolute()) if Path(args.python).exists() else args.python
        )
        script_results = [item for item in report["executions"] if item.get("kind") != "api_doctests"]
        api_results = [item for item in report["executions"] if item.get("kind") == "api_doctests"]
        scripts_passed = bool(script_results) and all(item["status"] == "passed" for item in script_results)
        api_passed = bool(api_results) and all(item["status"] == "passed" for item in api_results)
        report["example_execution_status"] = "passed" if scripts_passed else "failed_or_unexecuted"
        report["api_doctest_status"] = "passed" if api_passed else "failed_or_unexecuted"
        python_executed = scripts_passed and api_passed
        report["python_execution_status"] = "passed" if python_executed else "failed_or_unexecuted"
        report["language_receipts"] = verify_language_receipts(
            args.root, args.language_receipt_dir, report["required_languages"]
        )
        executed = (
            python_executed
            and bool(report["required_languages"])
            and all(item["status"] == "passed" for item in report["language_receipts"].values())
        )
        report["execution_status"] = "passed" if executed else "failed_or_unexecuted"
        passed = passed and executed
    report["current_python_content_complete"] = (
        report["current_structure_complete"] and report["python_execution_status"] == "passed"
    )
    report["current_content_complete"] = report["current_structure_complete"] and report["execution_status"] == "passed"
    report["full_scope_complete"] = report["full_scope_structure_complete"] and report["execution_status"] == "passed"
    report["complete"] = report["full_scope_complete"]
    if args.mode == "strict":
        passed = report["complete"]
    report["mode"] = args.mode
    report["gate_passed"] = passed
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
