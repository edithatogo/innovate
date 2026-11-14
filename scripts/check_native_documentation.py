"""Qualify the installed native module's actual exports and runtime docstrings.

This check imports the installed package deliberately. It complements the
import-free source inventory and Rust compiler checks; it does not attest that
the installed native binary was compiled from this checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import re
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("bindings/python/innovate")
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _sha256(path: Path) -> str:
    """Fingerprint source or an installed binary without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_checker():
    """Share the source contract's Google/NumPy parsing without importing Innovate."""
    path = Path(__file__).with_name("check_api_documentation.py")
    spec = importlib.util.spec_from_file_location("innovate_runtime_source_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_sections(doc: str, checker) -> dict[str, tuple[str, str]]:
    """End an indented Google section when ordinary unindented prose resumes."""
    sections = checker._sections(doc)
    for name, (style, body) in list(sections.items()):
        lines = body.splitlines()
        if style == "google" and lines and lines[0] != lines[0].lstrip():
            end = next(
                (index for index, line in enumerate(lines) if line.strip() and line == line.lstrip()), len(lines)
            )
            sections[name] = (style, textwrap.dedent("\n".join(lines[:end])).strip("\n"))
    return sections


def _scope(node: ast.AST):
    """Visit module-level declarations, including conditional ones, without bodies."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, DEFINITIONS):
            yield from _scope(child)


def _names(value: object, label: str) -> set[str]:
    """Require a finite, explicit export list rather than guessing dynamic intent."""
    if (
        not isinstance(value, (list, tuple))
        or not all(isinstance(name, str) and name.isidentifier() for name in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must contain unique identifier strings in a list or tuple")
    return set(value)


def _stub_inventory(source: str) -> tuple[dict[str, list[ast.AST]], set[str]]:
    """Collect source-declared callables without evaluating a type stub."""
    tree = ast.parse(source)
    nodes = list(_scope(tree))
    definitions: dict[str, list[ast.AST]] = {}
    for node in nodes:
        if isinstance(node, DEFINITIONS):
            definitions.setdefault(node.name, []).append(node)
    assignments = [
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]
    stores = [
        node
        for node in nodes
        if isinstance(node, ast.Name) and node.id == "__all__" and isinstance(node.ctx, ast.Store)
    ]
    mutations = [
        node
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "__all__"
    ]
    exports: set[str] = set()
    if stores or mutations:
        if len(assignments) != 1 or len(stores) != 1 or mutations or assignments[0] not in tree.body:
            raise ValueError("stub __all__ must be one unconditional literal assignment")
        exports = _names(ast.literal_eval(assignments[0].value), "stub __all__")
        # Named constants can be exported, but unresolved callable aliases cannot
        # silently evade an explicit source declaration.
        constants = {
            target.id
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        unresolved = exports - definitions.keys() - constants
        if unresolved:
            raise ValueError(f"stub exports have no source declaration: {sorted(unresolved)}")
    if not definitions:
        raise ValueError("native stub contains no callable declarations")
    return definitions, exports


def inspect_runtime(module: ModuleType, stub_source: str) -> dict[str, Any]:
    """Compare every visible/exported callable with its source stub and docs."""
    checker = _source_checker()
    declarations, stub_explicit = _stub_inventory(stub_source)
    gaps: list[dict[str, str]] = []

    def add(rule: str, symbol: str, detail: str) -> None:
        gaps.append({"rule": rule, "symbol": symbol, "detail": detail})

    namespace = vars(module)
    explicit: set[str] = set()
    if "__all__" in namespace:
        try:
            explicit = _names(namespace["__all__"], "runtime __all__")
        except ValueError as error:
            add("invalid_runtime_exports", "__all__", str(error))
    expected = {name for name in declarations if not name.startswith("_") or name in stub_explicit or name in explicit}
    # Union __dict__ and dir(): neither __all__ nor a custom __dir__ may hide a
    # public callable. PyO3 functions are builtins, so inspect.isfunction is wrong.
    visible = set(namespace) | set(dir(module)) | explicit | expected
    public = {name for name in visible if not name.startswith("_")} | explicit | expected
    actual: dict[str, object] = {}
    noncallable: list[str] = []
    for name in sorted(public):
        try:
            value = getattr(module, name)
        except AttributeError:
            add("missing_runtime_export", name, "The runtime module does not expose this declared name.")
            continue
        if callable(value):
            actual[name] = value
        else:
            noncallable.append(name)
            if name in expected:
                add(
                    "noncallable_runtime_export",
                    name,
                    "The stub declares a callable, but the runtime value is not callable.",
                )
    for name in sorted(actual.keys() - expected):
        add("undeclared_runtime_export", name, "Public runtime callable is absent from the stub's source declarations.")
    if not expected or not actual:
        add("empty_callable_inventory", module.__name__, "Both the stub and runtime must expose at least one callable.")

    def substantive(doc: object) -> bool:
        if not isinstance(doc, str) or not doc.strip():
            return False
        summary = inspect.cleandoc(doc).split("\n\n")[0].strip()
        signature_only = re.fullmatch(r"[\w.]+\s*\([^)]*\)(?:\s*->\s*[^\n]+)?", summary)
        placeholder = re.search(r"\b(?:TODO|TBD|FIXME|placeholder)\b", summary, re.IGNORECASE)
        return bool(checker._substantive(summary) and not signature_only and not placeholder)

    if not substantive(module.__doc__):
        add("missing_module_docstring", module.__name__, "Native module needs a substantive runtime description.")
    for name, value in sorted(actual.items()):
        doc = getattr(value, "__doc__", None)
        if not substantive(doc):
            add(
                "non_substantive_runtime_docstring",
                name,
                "A signature, missing text or placeholder is not documentation.",
            )
        sections = _runtime_sections(inspect.cleandoc(doc or ""), checker)
        functions = [
            node for node in declarations.get(name, []) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not functions:
            continue
        parameters = {
            argument.arg
            for node in functions
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                node.args.vararg,
                node.args.kwarg,
            )
            if argument is not None
        }
        documented = checker._entries(sections.get("parameters"))
        for parameter in sorted(parameters):
            if not checker._substantive(documented.get(parameter, "")):
                add("missing_runtime_parameter", name, parameter)
        for parameter in sorted(documented.keys() - parameters):
            add("unknown_runtime_parameter", name, parameter)
        if any(checker._needs_returns(node) for node in functions) and not checker._has_description(
            sections.get("returns")
        ):
            add("missing_runtime_returns", name, "Returned values need a substantive runtime description.")
        errors = checker._entries(sections.get("raises"))
        documented_errors = {
            key.rsplit(".", 1)[-1] for key, description in errors.items() if checker._substantive(description)
        }
        if not documented_errors:
            add("missing_runtime_raises", name, "Document the native transport's Python error conditions.")
        expected_errors = {
            key.rsplit(".", 1)[-1]
            for node in functions
            for key in checker._entries(_runtime_sections(ast.get_docstring(node) or "", checker).get("raises"))
        }
        for error in sorted(expected_errors - documented_errors):
            add("missing_runtime_exception", name, error)
    return {
        "stub_callables": sorted(expected),
        "runtime_callables": sorted(actual),
        "runtime_explicit_exports": sorted(explicit),
        "runtime_noncallable_exports": sorted(noncallable),
        "callable_count": len(actual),
        "gaps": gaps,
    }


def audit(root: Path) -> dict[str, Any]:
    """Inspect the installed extension and bind its Python source to this checkout."""
    root = root.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "native_documentation_complete": False,
        "complete": False,
        "installed_source_matches": False,
        "native_binary_source_binding_verified": False,
        "python_version": sys.version.split()[0],
        "module": "innovate._native",
        "gaps": [],
        "scope": (
            "Actual public module callables, including underscore names in __all__; runtime docstring structure "
            "and exact installed Python binding source. Noncallable exports are inventoried separately. "
            "This does not qualify Rust compiler docs, scientific accuracy, class-member runtime docs, "
            "or attest that the native binary was built from this checkout."
        ),
    }
    try:
        source = root / PACKAGE / "__init__.py"
        stub = root / PACKAGE / "_native.pyi"
        for path in (source, stub):
            current = root
            for part in path.relative_to(root).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError(f"checked source must not be a symlink: {current.relative_to(root)}")
            if not path.is_file():
                raise ValueError(f"checked source is missing: {path.relative_to(root)}")
        stub_source = stub.read_text(encoding="utf-8")
        _stub_inventory(stub_source)
        report["source_files"] = {path.relative_to(root).as_posix(): _sha256(path) for path in (source, stub)}
        package = importlib.import_module("innovate")
        native = importlib.import_module("innovate._native")
        if not isinstance(package, ModuleType) or not isinstance(native, ModuleType):
            raise ValueError("installed innovate and innovate._native must be modules")
        installed_source = Path(package.__file__)
        extension = Path(native.__file__)
        report["installed_binding"] = {"path": str(installed_source), "sha256": _sha256(installed_source)}
        report["installed_source_matches"] = (
            report["installed_binding"]["sha256"] == report["source_files"][source.relative_to(root).as_posix()]
        )
        if not report["installed_source_matches"]:
            report["gaps"].append(
                {
                    "rule": "installed_source_mismatch",
                    "symbol": "innovate",
                    "detail": "Installed Python binding bytes differ from the checked checkout.",
                }
            )
        loader = getattr(getattr(native, "__spec__", None), "loader", None)
        if not isinstance(loader, importlib.machinery.ExtensionFileLoader) or not any(
            str(extension).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            raise ValueError("innovate._native is not loaded by CPython's compiled-extension loader")
        report["installed_extension"] = {"path": str(extension), "sha256": _sha256(extension)}
        inspected = inspect_runtime(native, stub_source)
        report["gaps"].extend(inspected.pop("gaps"))
        report.update(inspected)
        if any(_sha256(root / path) != digest for path, digest in report["source_files"].items()):
            report["gaps"].append(
                {
                    "rule": "source_changed_during_audit",
                    "symbol": "innovate",
                    "detail": "Checked binding or stub changed during runtime inspection.",
                }
            )
        report["native_documentation_complete"] = not report["gaps"]
        report["status"] = "passed" if report["native_documentation_complete"] else "failed"
    except Exception as error:
        report["gaps"].append(
            {
                "rule": "invalid_runtime_evidence",
                "symbol": "innovate._native",
                "detail": f"{type(error).__name__}: {error}",
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Always write the current result, including import and malformed-input failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
