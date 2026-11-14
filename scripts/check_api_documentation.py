"""Inventory Python documentation without imports and qualify explicit contracts.

This checker measures structural coverage, not scientific or editorial quality.
Rust must be checked by its compiler; a Python or baseline pass never substitutes
for that evidence. The audit and ratchet modes deliberately retain every gap.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/contracts/api-documentation.json"
CALLABLES = (ast.FunctionDef, ast.AsyncFunctionDef)
DEFINITIONS = (*CALLABLES, ast.ClassDef)
SECTION_ALIASES = {"args": "parameters", "arguments": "parameters", "yields": "returns"}
INVENTORY_ERRORS = {
    "missing_source_root",
    "empty_source_root",
    "source_parse_error",
    "source_outside_root",
    "duplicate_symbol",
    "dynamic_exports",
    "unresolved_export",
    "imported_export",
}


def default_contract() -> dict[str, Any]:
    """Read the versioned policy maintained beside the site content."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _digest(node: ast.AST) -> str:
    """Hash syntax without line offsets so formatting cannot invalidate evidence."""
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _substantive(value: str) -> bool:
    """Reject empty text and obvious placeholders without asserting prose quality."""
    words = re.findall(r"[A-Za-z]+", value)
    return len(words) >= 3 and not re.match(
        r"\s*(?:todo|tbd|fixme|placeholder|documentation pending|to be documented)\b",
        value,
        re.IGNORECASE,
    )


def _sections(doc: str) -> dict[str, tuple[str, str]]:
    """Extract Google or NumPy sections, ending at every recognized heading."""
    lines = doc.splitlines()
    headings: list[tuple[int, int, str, str]] = []
    for index, line in enumerate(lines):
        if line and line == line.lstrip() and re.fullmatch(r"[A-Za-z][A-Za-z /-]*:", line):
            headings.append((index, index + 1, line[:-1].lower(), "google"))
        elif (
            line
            and line == line.lstrip()
            and index + 1 < len(lines)
            and re.fullmatch(r"-{3,}", lines[index + 1].strip())
        ):
            headings.append((index, index + 2, line.lower(), "numpy"))
    result = {}
    for position, (_, start, name, style) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        name = SECTION_ALIASES.get(name, name)
        result[name] = (style, textwrap.dedent("\n".join(lines[start:end])).strip("\n"))
    return result


def _entries(section: tuple[str, str] | None) -> dict[str, str]:
    """Read documented names and their descriptions from a structured section."""
    if section is None:
        return {}
    style, body = section
    entries: dict[str, list[str]] = {}
    active: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        if line == line.lstrip():
            if style == "google":
                match = re.fullmatch(r"([*\w., ]+)(?:\s*\([^:]*\))?\s*:\s*(.*)", line)
                if not match:
                    active = []
                    continue
                names, description = match.groups()
            else:
                names = line.partition(":")[0].strip()
                description = ""
            active = [name.strip().lstrip("*") for name in names.split(",")]
            for name in active:
                entries[name] = [description]
        else:
            for name in active:
                entries[name].append(line.strip())
    return {name: " ".join(parts).strip() for name, parts in entries.items()}


def _has_description(section: tuple[str, str] | None) -> bool:
    """Require explanatory content beyond a section heading or return type."""
    if section is None:
        return False
    style, body = section
    if style == "google" and ":" in body.split("\n")[0]:
        body = body.partition(":")[2]
    elif style == "numpy":
        body = "\n".join(line.strip() for line in body.splitlines() if line != line.lstrip())
    return _substantive(body)


def _scoped_nodes(node: ast.AST):
    """Walk a callable body without attributing nested definitions to its parent."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, DEFINITIONS):
            continue
        yield child
        yield from _scoped_nodes(child)


def _name(node: ast.AST | None) -> str | None:
    """Resolve a simple syntactic name without importing or evaluating code."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _needs_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Include annotated results and actual value returns, excluding constructors."""
    if isinstance(node.returns, ast.Constant) and node.returns.value is None:
        return False
    if node.returns is not None:
        return True
    return any(
        isinstance(child, (ast.Yield, ast.YieldFrom))
        or (
            isinstance(child, ast.Return)
            and child.value is not None
            and not (isinstance(child.value, ast.Constant) and child.value.value is None)
        )
        for child in _scoped_nodes(node)
    )


def _validate_contract(contract: dict[str, Any]) -> None:
    """Fail closed on unsupported policies instead of silently reducing coverage."""
    if (
        not isinstance(contract, dict)
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != 1
    ):
        raise ValueError("unsupported API documentation contract schema")
    roots = contract.get("python_roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(p, str) and p for p in roots):
        raise ValueError("python_roots must be a nonempty list of paths")
    if len(roots) != len(set(roots)) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in roots):
        raise ValueError("python_roots must be unique relative paths without traversal")
    if contract.get("python_extensions") != [".py", ".pyi"]:
        raise ValueError("both Python implementations and type stubs must be inventoried")
    rules = contract.get("public_contract", {})
    if not isinstance(rules, dict) or any(
        rules.get(name) is not True
        for name in ("parameters", "returns", "raises", "attributes", "unresolved_exports_fail")
    ):
        raise ValueError("the v1 public documentation rules cannot be disabled")


def audit(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Return source-derived gaps, preserving incomplete and unexecuted states."""
    _validate_contract(contract)
    root = root.resolve()
    symbols: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    seen: set[str] = set()

    def gap(symbol: dict[str, Any], rule: str, detail: str = "") -> None:
        gaps.append(
            {
                "id": f"{symbol['id']}::{rule}" + (f":{detail}" if detail else ""),
                "symbol": symbol["id"],
                "rule": rule,
                "detail": detail,
                "source_sha256": symbol["source_sha256"],
            }
        )

    def visit(node: ast.AST, path: str, name: str, exposure: str, parent_kind: str = "") -> None:
        kind = "module" if isinstance(node, ast.Module) else "class" if isinstance(node, ast.ClassDef) else "callable"
        symbol = {
            "id": f"{path}::{name}",
            "name": name,
            "path": path,
            "line": getattr(node, "lineno", 1),
            "kind": kind,
            "exposure": exposure,
            "source_sha256": _digest(node),
        }
        if symbol["id"] in seen:
            gap(symbol, "duplicate_symbol")
        seen.add(symbol["id"])
        symbols.append(symbol)
        doc = ast.get_docstring(node) or ""
        if not doc.strip():
            gap(symbol, "missing_docstring")
        elif not _substantive(doc.split("\n\n")[0]):
            gap(symbol, "placeholder_docstring")
        sections = _sections(doc)
        if exposure == "public" and isinstance(node, CALLABLES):
            parameters = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            if node.args.vararg:
                parameters.append(node.args.vararg.arg)
            if node.args.kwarg:
                parameters.append(node.args.kwarg.arg)
            static = any(_name(decorator) == "staticmethod" for decorator in node.decorator_list)
            if parent_kind == "class" and not static and parameters and parameters[0] in {"self", "cls"}:
                parameters.pop(0)
            documented = _entries(sections.get("parameters"))
            for parameter in sorted(parameters):
                if not _substantive(documented.get(parameter, "")):
                    gap(symbol, "missing_parameter", parameter)
            for parameter in sorted(set(documented) - set(parameters)):
                gap(symbol, "unknown_parameter", parameter)
            if _needs_returns(node) and not _has_description(sections.get("returns")):
                gap(symbol, "missing_returns")
            exceptions = _entries(sections.get("raises"))
            if not any(_substantive(description) for description in exceptions.values()):
                gap(symbol, "missing_raises")
            documented_exceptions = {
                name.rsplit(".", 1)[-1] for name, description in exceptions.items() if _substantive(description)
            }
            for child in _scoped_nodes(node):
                if isinstance(child, ast.Raise):
                    exception = _name(child.exc.func if isinstance(child.exc, ast.Call) else child.exc)
                    if exception and exception.rsplit(".", 1)[-1] not in documented_exceptions:
                        gap(symbol, "missing_exception", exception)
        if exposure == "public" and isinstance(node, ast.ClassDef):
            attributes = _entries(sections.get("attributes"))
            for child in node.body:
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and not child.target.id.startswith("_")
                ):
                    if not _substantive(attributes.get(child.target.id, "")):
                        gap(symbol, "missing_attribute", child.target.id)
        explicit_exports = _check_exports(node, symbol, gap) if isinstance(node, ast.Module) else set()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFINITIONS):
                special = (
                    parent_kind != "callable"
                    and kind == "class"
                    and child.name.startswith("__")
                    and child.name.endswith("__")
                )
                child_exposure = (
                    "nested"
                    if kind == "callable" or exposure == "nested"
                    else "public"
                    if child.name in explicit_exports
                    else "private"
                    if exposure == "private" or (child.name.startswith("_") and not special)
                    else "public"
                )
                visit(child, path, f"{name}.{child.name}", child_exposure, kind)
            else:
                visit_container(child, path, name, exposure, kind)

    def visit_container(node: ast.AST, path: str, name: str, exposure: str, kind: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFINITIONS):
                child_exposure = (
                    "nested"
                    if kind == "callable"
                    else "private"
                    if exposure == "private" or child.name.startswith("_")
                    else "public"
                )
                visit(child, path, f"{name}.{child.name}", child_exposure, kind)
            else:
                visit_container(child, path, name, exposure, kind)

    for package in contract["python_roots"]:
        directory = root / package
        location = {"id": package, "source_sha256": ""}
        if not directory.is_dir():
            gap(location, "missing_source_root")
            continue
        paths = sorted(path for path in directory.rglob("*") if path.suffix in contract["python_extensions"])
        if not paths:
            gap(location, "empty_source_root")
        for path in paths:
            relative = path.relative_to(root).as_posix()
            location = {"id": relative, "source_sha256": ""}
            if not path.resolve().is_relative_to(root):
                gap(location, "source_outside_root")
                continue
            try:
                raw = path.read_bytes()
                tree = ast.parse(raw, filename=relative)
            except (OSError, SyntaxError, ValueError) as error:
                gap(location, "source_parse_error", type(error).__name__)
                continue
            files[relative] = hashlib.sha256(raw).hexdigest()
            pieces = list(path.relative_to(directory.parent).with_suffix("").parts)
            if pieces[-1] == "__init__":
                pieces.pop()
            module = ".".join(pieces)
            exposure = "private" if any(piece.startswith("_") for piece in pieces) else "public"
            visit(tree, relative, module, exposure)
    # Multiple explicit raises of one exception are one obligation, not duplicates.
    gaps = sorted({item["id"]: item for item in gaps}.values(), key=lambda item: item["id"])
    inventory_valid = not any(item["rule"] in INVENTORY_ERRORS for item in gaps)
    return {
        "schema_version": 1,
        "scope": {"python_roots": contract["python_roots"], "python_extensions": contract["python_extensions"]},
        "inventory_valid": inventory_valid,
        "python_complete": inventory_valid and not gaps,
        "rust_status": "not_executed",
        "complete": False,
        "qualification": (
            "Structural Python documentation only; Rust, executable examples "
            "and scientific prose require separate evidence."
        ),
        "counts": {
            "files": len(files),
            "symbols": len(symbols),
            "gaps": len(gaps),
            "public_symbols": sum(item["exposure"] == "public" for item in symbols),
        },
        "files": files,
        "symbols": sorted(symbols, key=lambda item: item["id"]),
        "gaps": gaps,
    }


def _check_exports(node: ast.Module, symbol: dict[str, Any], gap) -> set[str]:
    """Reject unresolved or dynamically mutated __all__ surfaces."""
    exports = None
    definitions = {child.name for child in node.body if isinstance(child, DEFINITIONS)}
    bindings = set(definitions)
    imported = set()
    assignments = []
    for child in node.body:
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in child.names)
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    bindings.add(target.id)
                    if target.id == "__all__":
                        assignments.append(child)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            bindings.add(child.target.id)
            if child.target.id == "__all__":
                assignments.append(child)
    if len(assignments) == 1:
        try:
            value = ast.literal_eval(assignments[0].value)
            if not isinstance(value, (list, tuple)) or not all(isinstance(name, str) for name in value):
                raise ValueError("non-string exports")
            exports = value
            if len(value) != len(set(value)):
                gap(symbol, "dynamic_exports", "duplicate_export_names")
        except ValueError, TypeError, SyntaxError:
            gap(symbol, "dynamic_exports", "nonliteral_assignment")
    elif len(assignments) > 1:
        gap(symbol, "dynamic_exports", "multiple_assignments")
    allowed = {
        id(target)
        for assignment in assignments
        for target in (assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target])
    }
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "__all__" and id(child) not in allowed:
            gap(symbol, "dynamic_exports", "export_mutation_or_reference")
    for name in exports or []:
        if name in imported:
            gap(symbol, "imported_export", name)
        elif name not in bindings:
            gap(symbol, "unresolved_export", name)
    return set(exports or [])


def make_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """Create a reviewable baseline value; this function never writes or accepts it."""
    return {
        "schema_version": 1,
        "scope": report["scope"],
        "symbols": [symbol["id"] for symbol in report["symbols"]],
        "gaps": [{"id": gap["id"], "source_sha256": gap["source_sha256"]} for gap in report["gaps"]],
    }


def evaluate_ratchet(report: dict[str, Any], baseline: Any) -> dict[str, Any]:
    """Check reviewed gap identities/digests without ever granting completeness."""
    result = {
        "passed": False,
        "errors": [],
        "new_gaps": [],
        "changed_gaps": [],
        "stale_gaps": [],
        "removed_symbols": [],
    }
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema_version") != 1
        or baseline.get("scope") != report["scope"]
    ):
        result["errors"].append("unsupported baseline schema or mismatched scope")
        return result
    entries, symbols = baseline.get("gaps"), baseline.get("symbols")
    if (
        not isinstance(entries, list)
        or not isinstance(symbols, list)
        or not all(isinstance(item, str) for item in symbols)
    ):
        result["errors"].append("baseline gaps and symbols must be lists")
        return result
    if len(symbols) != len(set(symbols)) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("id"), str)
        or not isinstance(item.get("source_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", item["source_sha256"])
        for item in entries
    ):
        result["errors"].append("invalid baseline symbol or gap entries")
        return result
    accepted = {item["id"]: item["source_sha256"] for item in entries}
    if len(accepted) != len(entries):
        result["errors"].append("duplicate baseline gap IDs")
        return result
    actual = {gap["id"]: gap["source_sha256"] for gap in report["gaps"]}
    result["new_gaps"] = sorted(actual.keys() - accepted.keys())
    result["changed_gaps"] = sorted(key for key in actual.keys() & accepted.keys() if actual[key] != accepted[key])
    result["stale_gaps"] = sorted(accepted.keys() - actual.keys())
    result["removed_symbols"] = sorted(set(symbols) - {symbol["id"] for symbol in report["symbols"]})
    if not report["inventory_valid"]:
        result["errors"].append("inventory is incomplete or unresolved")
    result["passed"] = not any(
        result[name] for name in ("errors", "new_gaps", "changed_gaps", "stale_gaps", "removed_symbols")
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run strict qualification, explicit audit, or a reviewed non-regression gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--mode", choices=("strict", "audit", "ratchet"), default="strict")
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs/contracts/api-baseline.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root, json.loads(args.contract.read_text(encoding="utf-8")))
        if args.mode == "ratchet":
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            report["ratchet"] = evaluate_ratchet(report, baseline)
            passed = report["ratchet"]["passed"]
        else:
            passed = report["inventory_valid"] if args.mode == "audit" else report["python_complete"]
        report["mode"] = args.mode
        report["command_passed"] = passed
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
            print(
                f"API docs {args.mode}: command_passed={passed}; python_complete={report['python_complete']}; "
                f"gaps={report['counts']['gaps']}; Rust=not_executed; complete=False"
            )
        else:
            print(encoded, end="")
        return 0 if passed else 1
    except (OSError, ValueError, TypeError) as error:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "command_passed": False,
            "inventory_valid": False,
            "python_complete": False,
            "rust_status": "not_executed",
            "complete": False,
            "errors": [{"code": "invalid_input", "detail": str(error)}],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        print(f"API documentation qualification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
