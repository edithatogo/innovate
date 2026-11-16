"""Reject misleading notebook qualification using only the standard library.

The source fixtures and mocked runtime below test the checker, not Innovate,
marimo, a native binary, browser execution or the scientific interpretation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_learning_notebooks.py"
SPEC = importlib.util.spec_from_file_location("learning_notebooks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class LearningNotebookTests(unittest.TestCase):
    """Use independent source/data inventories and explicit failure injections."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset_path = "docs/datasets/fixture/manifest.json"
        self.entry = {
            "id": "public-data-fit",
            "source": "notebooks/public_data_fit.py",
            "example": "examples/native_public_data.py",
            "dataset": self.dataset_path,
            "default_cutoff": 2015,
            "alternate_cutoff": 2018,
            "result_variable": "comparison",
            "control_variable": "cutoff",
        }
        self.contract = {
            "schema_version": 1,
            "environment": {
                "python": "3.15.2",
                "marimo": "0.25.2",
                "polars": "2.1.0",
                "innovate": "0.7.0a1",
                "project": "notebooks/pyproject.toml",
                "lock": "notebooks/uv.lock",
            },
            "execution_mode": "native_python",
            "browser_wasm_qualified": False,
            "static_export_published": False,
            "notebooks": [self.entry],
        }
        self.content = {"examples": {self.entry["example"]: {"status": "runnable"}}}
        self.project = (
            '[project]\nname = "fixture-notebooks"\nversion = "0.1.0"\n'
            'requires-python = ">=3.15.2,<3.16"\n'
            'dependencies = ["innovate", "marimo==0.25.2"]\n\n'
            '[tool.uv.sources]\ninnovate = { path = "..", editable = true }\n\n'
            "[tool.uv]\npackage = false\n"
        )
        self.lock = (
            'version = 1\nrevision = 5\nrequires-python = ">=3.15.2, <3.16"\n\n'
            '[[package]]\nname = "marimo"\nversion = "0.25.2"\n'
            'source = { registry = "https://pypi.org/simple" }\n\n'
            '[[package]]\nname = "polars"\nversion = "2.1.0"\n'
            'source = { registry = "https://pypi.org/simple" }\n\n'
            '[[package]]\nname = "innovate"\nversion = "0.7.0a1"\nsource = { editable = "../" }\n'
        )
        self.write("scripts/check_learning_notebooks.py", SCRIPT.read_text(encoding="utf-8"))
        self.write(
            "pyproject.toml",
            (
                '[project]\nname = "innovate"\nversion = "0.7.0a1"\n'
                'requires-python = ">=3.15.2,<3.16"\ndependencies = ["polars==2.1.0"]\n'
            ),
        )
        self.write("notebooks/pyproject.toml", self.project)
        self.write("notebooks/uv.lock", self.lock)
        self.write(self.entry["source"], '"""A source-only notebook fixture."""\nraise RuntimeError("must not run")\n')
        self.write(
            self.entry["example"], '"""A source-only registered example."""\nraise RuntimeError("must not run")\n'
        )
        raw = self.write("docs/datasets/fixture/raw.json", '[{"year": 2015, "value": 50}]\n')
        csv = self.write("docs/datasets/fixture/observations.csv", "year,time,observed\n2015,25,50\n")
        self.dataset = {
            "schema_version": 1,
            "license": "CC-BY-4.0",
            "unit": "percent_of_population",
            "time_unit": "year",
            "attribution": "An independently authored synthetic fixture; no empirical observation is represented.",
            "observation": "A current-use percentage, not cumulative first adoption.",
            "limitations": ["Only a metadata/checksum test; no scientific inference."],
            "files": {
                "raw.json": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "observations.csv": hashlib.sha256(csv.read_bytes()).hexdigest(),
            },
        }
        self.write_inputs()

    def write(self, relative: str, contents: str) -> Path:
        """Write inside this test's temporary repository."""
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
        return destination

    def write_inputs(self) -> None:
        """Serialize the independent declarations, leaving source files unchanged."""
        self.write(str(checker.CONTRACT), json.dumps(self.contract))
        self.write("docs/contracts/content-quality.json", json.dumps(self.content))
        self.write(self.dataset_path, json.dumps(self.dataset))

    def call_main(self) -> tuple[int, dict]:
        """Run the default CLI against a fixture without touching shared reports."""
        stream = io.StringIO()
        with patch.object(checker, "ROOT", self.root), patch.object(sys, "argv", [str(SCRIPT), "--json"]):
            with redirect_stdout(stream):
                status = checker.main()
        return status, json.loads(stream.getvalue())

    def test_source_validation_is_deterministic_and_fingerprints_inputs_without_execution(self) -> None:
        first = checker.validate(self.root, self.contract)
        self.assertEqual(checker.validate(self.root, self.contract), first)
        for source in (
            str(checker.CONTRACT),
            "docs/contracts/content-quality.json",
            "scripts/check_learning_notebooks.py",
            "pyproject.toml",
            "notebooks/pyproject.toml",
            "notebooks/uv.lock",
            self.entry["source"],
            self.entry["example"],
            self.dataset_path,
            "docs/datasets/fixture/raw.json",
            "docs/datasets/fixture/observations.csv",
        ):
            self.assertEqual(first[source], hashlib.sha256((self.root / source).read_bytes()).hexdigest())
        self.write(self.entry["example"], '"""A changed but syntactically valid source fixture."""\n')
        changed = checker.validate(self.root, self.contract)
        self.assertNotEqual(first[self.entry["example"]], changed[self.entry["example"]])

    def test_schema_and_false_delivery_claims_fail_closed(self) -> None:
        for field, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("execution_mode", "browser_wasm"),
            ("browser_wasm_qualified", True),
            ("browser_wasm_qualified", 0),
            ("static_export_published", True),
            ("static_export_published", 0),
            ("scientific_acceptance", True),
            ("notebooks", []),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(self.contract)
                broken[field] = value
                with self.assertRaises(ValueError):
                    checker.validate(self.root, broken)

    def test_invalid_and_unsafe_source_fields_fail(self) -> None:
        for field in ("source", "example", "dataset"):
            for value in (123, None, "../outside.py", "/tmp/outside.py", "notebooks/../outside.py"):
                with self.subTest(field=field, value=value):
                    broken = copy.deepcopy(self.contract)
                    broken["notebooks"][0][field] = value
                    with self.assertRaises(ValueError):
                        checker.validate(self.root, broken)

    def test_symlinked_source_and_dataset_paths_fail(self) -> None:
        for relative in (self.entry["source"], self.entry["example"], "docs/datasets/fixture/raw.json"):
            with self.subTest(path=relative):
                original = self.root / relative
                data = original.read_bytes()
                backup = self.write("outside-fixture.txt", data.decode())
                original.unlink()
                original.symlink_to(backup)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    checker.validate(self.root, self.contract)
                original.unlink()
                original.write_bytes(data)

    def test_missing_and_unlisted_notebooks_fail(self) -> None:
        source = self.root / self.entry["source"]
        source.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            checker.validate(self.root, self.contract)
        self.write(self.entry["source"], '"""Restored fixture."""\n')
        self.write("notebooks/unregistered.py", '"""An unregistered notebook."""\n')
        with self.assertRaisesRegex(ValueError, "inventory"):
            checker.validate(self.root, self.contract)

    def test_duplicate_notebook_ids_and_sources_fail(self) -> None:
        for change in ({}, {"id": "another-notebook"}):
            with self.subTest(change=change):
                broken = copy.deepcopy(self.contract)
                second = {**broken["notebooks"][0], **change}
                broken["notebooks"].append(second)
                with self.assertRaises(ValueError):
                    checker.validate(self.root, broken)

    def test_notebooks_require_registered_runnable_examples(self) -> None:
        broken = copy.deepcopy(self.contract)
        broken["notebooks"][0]["example"] = "examples/unregistered.py"
        with self.assertRaisesRegex(ValueError, "registered runnable"):
            checker.validate(self.root, broken)
        self.content["examples"][self.entry["example"]]["status"] = "historical"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "registered runnable"):
            checker.validate(self.root, self.contract)

    def test_changed_control_must_be_distinct_and_integer_in_range(self) -> None:
        for field, value in (
            ("alternate_cutoff", 2015),
            ("alternate_cutoff", True),
            ("default_cutoff", 2009),
            ("alternate_cutoff", 2021),
            ("default_cutoff", 2015.0),
            ("control_variable", "unread_control"),
            ("result_variable", "uninspected_result"),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(self.contract)
                broken["notebooks"][0][field] = value
                with self.assertRaises(ValueError):
                    checker.validate(self.root, broken)

    def test_invalid_source_syntax_cannot_pass_structure_validation(self) -> None:
        for relative in (self.entry["source"], self.entry["example"]):
            with self.subTest(path=relative):
                previous = (self.root / relative).read_text()
                self.write(relative, "def broken(:\n")
                with self.assertRaises((SyntaxError, ValueError)):
                    checker.validate(self.root, self.contract)
                self.write(relative, previous)

    def test_dataset_semantics_and_type_exact_schema_are_required(self) -> None:
        for field, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("license", "unknown"),
            ("unit", "cumulative_adopters"),
            ("time_unit", "month"),
            ("attribution", ""),
            ("observation", ""),
            ("limitations", []),
        ):
            with self.subTest(field=field, value=value):
                broken = {**self.dataset, field: value}
                self.write(self.dataset_path, json.dumps(broken))
                with self.assertRaises(ValueError):
                    checker.validate(self.root, self.contract)

    def test_data_corruption_and_missing_dataset_files_fail(self) -> None:
        for name in ("raw.json", "observations.csv"):
            with self.subTest(path=name):
                path = self.root / "docs/datasets/fixture" / name
                original = path.read_bytes()
                path.write_bytes(original + b"altered observation\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    checker.validate(self.root, self.contract)
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    checker.validate(self.root, self.contract)
                path.write_bytes(original)

    def test_dataset_file_inventory_cannot_hide_omissions_or_escape(self) -> None:
        for files in (
            {"raw.json": self.dataset["files"]["raw.json"]},
            {**self.dataset["files"], "../outside.json": "a" * 64},
        ):
            with self.subTest(files=list(files)):
                self.write(self.dataset_path, json.dumps({**self.dataset, "files": files}))
                with self.assertRaises(ValueError):
                    checker.validate(self.root, self.contract)

    def test_project_and_lock_pins_must_be_parsed_not_found_in_comments(self) -> None:
        for project in (
            self.project.replace('"marimo==0.25.2"', '"marimo==0.1.0"'),
            self.project.replace('"marimo==0.25.2"', '"marimo==0.1.0"') + "# marimo==0.25.2\n",
            "[project\n# marimo==0.25.2\n",
        ):
            with self.subTest(project=project):
                self.write("notebooks/pyproject.toml", project)
                with self.assertRaises(ValueError):
                    checker.validate(self.root, self.contract)
        self.write("notebooks/pyproject.toml", self.project)
        for lock in (self.lock.replace('version = "2.1.0"', 'version = "2.0.0"'), "[[package\n"):
            with self.subTest(lock=lock):
                self.write("notebooks/uv.lock", lock)
                with self.assertRaises(ValueError):
                    checker.validate(self.root, self.contract)

    def test_default_cli_preserves_execution_receipt_and_reports_no_execution(self) -> None:
        execution = self.write(str(checker.REPORT), '{"prior_execution":"preserve exact bytes"}\n')
        original = execution.read_bytes()
        with patch.object(
            checker, "execute", side_effect=AssertionError("source mode executed native code")
        ) as execute:
            status, report = self.call_main()
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "passed")
        for field in ("executed", "browser_wasm_qualified", "static_export_published", "scientific_acceptance"):
            self.assertIs(report[field], False)
        execute.assert_not_called()
        self.assertEqual(execution.read_bytes(), original)
        structure = execution.with_name("structure.json")
        self.assertEqual(json.loads(structure.read_text()), report)

    def test_bad_source_replaces_structure_receipt_with_failure_preserving_execution(self) -> None:
        execution = self.write(str(checker.REPORT), '{"prior_execution":"preserve exact bytes"}\n')
        original = execution.read_bytes()
        structure = execution.with_name("structure.json")
        structure.write_text('{"status":"passed"}\n')
        self.write(self.entry["source"], "def broken(:\n")
        status, report = self.call_main()
        self.assertEqual(status, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIs(report["executed"], False)
        self.assertIn("error", report)
        self.assertEqual(json.loads(structure.read_text()), report)
        self.assertEqual(execution.read_bytes(), original)

    def test_export_failure_cannot_be_hidden_by_an_existing_html_file(self) -> None:
        output = self.write("exports/lesson.html", "<!doctype html><p>Earlier export, not a successful new run.</p>")
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "export failed"):
            checker.verify_export(1, output)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(checker.verify_export(0, output), hashlib.sha256(original).hexdigest())
        output.write_bytes(b"")
        with self.assertRaises(ValueError):
            checker.verify_export(0, output)
        output.unlink()
        with self.assertRaises(ValueError):
            checker.verify_export(0, output)

    def stub_execution(self, app_result, example_result):
        """Stub external dependencies to test rejection logic, never native results."""
        contract = copy.deepcopy(self.contract)
        contract["environment"]["python"] = ".".join(map(str, sys.version_info[:3]))
        binding = self.write("bindings/python/innovate/__init__.py", '"""Synthetic binding identity."""\n')
        binary = self.write("fixture-native.so", "Synthetic file: not a compiled library.")
        module = ModuleType("innovate")
        module.__file__ = str(binding)
        native = ModuleType("innovate._native")
        native.__file__ = str(binary)
        module._native = native

        def run(defs=None):
            cutoff = self.entry["default_cutoff"] if defs is None else defs["cutoff"].value
            return (), {"comparison": app_result(cutoff)}

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, {"innovate": module, "innovate._native": native}))
        stack.enter_context(
            patch.object(checker.importlib.metadata, "version", side_effect=contract["environment"].get)
        )
        stack.enter_context(
            patch.object(
                checker,
                "_module",
                side_effect=[
                    SimpleNamespace(app=SimpleNamespace(run=run)),
                    SimpleNamespace(run_comparison=example_result),
                ],
            )
        )
        subprocess_run = stack.enter_context(
            patch.object(checker.subprocess, "run", return_value=SimpleNamespace(returncode=0))
        )
        return contract, subprocess_run

    def test_changed_control_must_change_the_calculation_even_if_both_calls_match_example(self) -> None:
        def fixed(cutoff):
            return {"cutoff_year": cutoff, "predictions": [{"prediction": 42.0}], "holdout_rows": 1, "scores": []}

        contract, calls = self.stub_execution(fixed, fixed)
        before = sys.path.copy()
        with self.assertRaisesRegex(ValueError, "changed control"):
            checker.execute(self.root, contract, self.root / "exports")
        self.assertEqual(sys.path, before)
        self.assertEqual(calls.call_count, 1, "A failed calculation must stop before export")

    def test_notebook_example_parity_failure_stops_before_export(self) -> None:
        def notebook(cutoff):
            return {"cutoff_year": cutoff, "predictions": [{"prediction": 41.0}]}

        def example(cutoff):
            return {"cutoff_year": cutoff, "predictions": [{"prediction": 42.0}]}

        contract, calls = self.stub_execution(notebook, example)
        with self.assertRaisesRegex(ValueError, "differ"):
            checker.execute(self.root, contract, self.root / "exports")
        self.assertEqual(calls.call_count, 1)


if __name__ == "__main__":
    unittest.main()
