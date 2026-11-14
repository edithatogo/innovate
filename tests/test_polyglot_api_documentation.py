"""Mutation tests for the bounded, non-executing Julia/R documentation gate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_polyglot_api_documentation.py"


def load_checker():
    """Import the checker, never either language's source implementation."""
    spec = importlib.util.spec_from_file_location("polyglot_documentation_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PolyglotDocumentationTests(unittest.TestCase):
    """Exercise evidence drift and misleading documentation independently."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checker = load_checker()
        self.write(self.checker.CHECKER, SCRIPT.read_text(encoding="utf-8"))
        self.write(
            self.checker.WORKFLOW,
            "\n".join(
                "- run: " + command
                for command in [*self.checker.COMMANDS.values(), *self.checker.DOCUMENTATION_COMMANDS.values()]
            ),
        )
        for path in self.checker.RUNNERS.values():
            self.write(path, "# Structural fixture runner; this file is never executed.\n")
        self.checker.REVIEWED_JULIA_EXAMPLES = ("measure",)
        self.write("bindings/julia/Project.toml", '[compat]\njulia = "1.13"\nJSON = "1.13.4"\n')
        self.write("bindings/r/DESCRIPTION", "Package: innovate.R\nImports: jsonlite (>= 2.0.0)\n")
        self.write(
            self.checker.JULIA,
            '''
            module Innovate
            export measure
            """
                measure(value; scale=1)

            Scale the caller's measurement in its original units.

            # Arguments
            - `value`: The supplied measurement in its original units.
            - `scale`: A dimensionless multiplier defaulting to one.

            # Returns
            The scaled measurement in the same units.

            # Throws
            Arithmetic failures propagate from the input number type.

            # Examples
            ```julia
            @assert measure(2) == 2
            ```
            """
            measure(value; scale=1) = value * scale
            end
            ''',
        )
        self.write("bindings/r/NAMESPACE", "useDynLib(innovate.R, .registration = TRUE)\nexport(measure)\n")
        self.write(
            self.checker.R_SOURCE,
            """
            #' Scale a measurement in its original units.
            #'
            #' @param value The input measurement in its original units.
            #' @param scale A dimensionless multiplier defaulting to one.
            #' @return The scaled measurement in the original units.
            #' @section Errors: Arithmetic failures propagate from the number type.
            measure <- function(value, scale = 1L) value * scale
            """,
        )
        self.manual = "bindings/r/man/measure.Rd"
        self.checker.REVIEWED_R_EXAMPLES = (self.manual,)
        self.write(
            self.manual,
            r"""
            \name{measure}
            \alias{measure}
            \title{Scale a measurement}
            \description{Multiply a measurement by a dimensionless scale factor.}
            \usage{measure(value, scale = 1L)}
            \arguments{
            \item{value}{Input measurement in its original units.}
            \item{scale}{Dimensionless multiplier defaulting to one.}
            }
            \value{Scaled measurement in its original units.}
            \examples{
            stopifnot(measure(2) == 2)
            }
            """,
        )

    def write(self, relative, content):
        """Create one explicit fixture input with deterministic newlines."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")

    def replace(self, relative, old, new):
        """Assert that a mutation affects real fixture content."""
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def inventory(self):
        """Inspect only static fixture bytes through the public callable API."""
        return self.checker.build_inventory(self.root)

    def record(self):
        """Write expected metadata in the fixture, simulating explicit refresh."""
        path = self.root / self.checker.INVENTORY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.checker.render_inventory(self.inventory()), encoding="utf-8")

    def test_structure_pass_is_not_runtime_or_full_documentation_acceptance(self):
        report = self.inventory()
        self.assertTrue(report["structural_complete"])
        self.assertFalse(report["documentation_complete"])
        self.assertEqual(report["runtime_qualification"], "not_run_by_this_checker")
        self.assertEqual(report["counts"]["julia_exported_symbols"], 1)
        self.assertEqual(report["counts"]["r_exported_functions"], 1)
        self.assertIn("not_qualified", report["rendered_reference"]["julia"])
        self.assertEqual(report["counts"]["julia_example_blocks"], 1)
        self.assertEqual(report["counts"]["r_example_blocks"], 1)
        self.assertEqual(report["counts"]["julia_example_assertions"], 1)
        self.assertEqual(report["counts"]["r_example_assertions"], 1)
        self.assertEqual(report["counts"]["r_all_manual_pages"], 1)
        self.assertEqual(report["hosted_documentation_checks"]["status"], "configured_pending_execution")
        self.assertEqual(report["hosted_documentation_checks"]["local_execution"], "not_run")

    def test_deterministic_inventory_and_read_only_check(self):
        first = self.checker.render_inventory(self.inventory())
        self.assertEqual(first, self.checker.render_inventory(self.inventory()))
        self.record()
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.checker.check_inventory(self.root), [])
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_or_forged_inventory_fails(self):
        self.assertTrue(self.checker.check_inventory(self.root))
        self.record()
        path = self.root / self.checker.INVENTORY
        recorded = json.loads(path.read_text())
        recorded["documentation_complete"] = True
        path.write_text(self.checker.render_inventory(recorded))
        self.assertIn("stale or modified", self.checker.check_inventory(self.root)[0])

    def test_source_change_invalidates_recorded_identity(self):
        self.record()
        self.replace(self.checker.JULIA, "The scaled measurement", "The rescaled measurement")
        self.assertTrue(self.checker.check_inventory(self.root))

    def test_source_is_not_executed(self):
        with (self.root / self.checker.JULIA).open("a") as file:
            file.write('error("must never execute this file")\n')
        with (self.root / self.checker.R_SOURCE).open("a") as file:
            file.write('stop("must never execute this file")\n')
        self.assertTrue(self.inventory()["structural_complete"])

    def test_undefined_julia_export_is_not_dropped(self):
        self.replace(self.checker.JULIA, "export measure", "export measure, absent")
        with self.assertRaisesRegex(ValueError, "absent.*source definition"):
            self.inventory()

    def test_comment_does_not_document_julia_export(self):
        path = self.root / self.checker.JULIA
        text = path.read_text()
        first = text.index('"""')
        last = text.index('"""', first + 3) + 3
        path.write_text(text[:first] + "# This comment is not an attached docstring.\n" + text[last:])
        with self.assertRaisesRegex(ValueError, "missing attached Julia docstring"):
            self.inventory()

    def test_undocumented_julia_parameter_and_default_drift_fail(self):
        self.replace(
            self.checker.JULIA,
            "- `scale`: A dimensionless multiplier defaulting to one.",
            "Scale is mentioned in this unstructured paragraph only.",
        )
        with self.assertRaisesRegex(ValueError, "missing=\\['scale'\\]"):
            self.inventory()

    def test_julia_documented_signature_must_match_defaults(self):
        self.replace(self.checker.JULIA, "measure(value; scale=1) =", "measure(value; scale=2) =")
        with self.assertRaisesRegex(ValueError, "documented signature differs"):
            self.inventory()

    def test_julia_fenced_examples_cannot_supply_returns_section(self):
        self.replace(
            self.checker.JULIA,
            "# Returns\nThe scaled measurement in the same units.",
            "```julia\n# Returns\nThe scaled measurement in the same units.\n```",
        )
        with self.assertRaisesRegex(ValueError, "missing substantive returns"):
            self.inventory()

    def test_dynamic_or_additional_julia_methods_require_review(self):
        with (self.root / self.checker.JULIA).open("a") as file:
            file.write("measure(value::Int; scale=1) = value * scale\n")
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.inventory()
        self.replace(self.checker.JULIA, "measure(value::Int; scale=1) = value * scale", "@eval export hidden")
        with self.assertRaisesRegex(ValueError, "dynamic Julia"):
            self.inventory()

    def test_empty_export_scopes_fail(self):
        self.replace(self.checker.JULIA, "export measure", "")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.inventory()

    def test_julia_conditional_public_definition_is_not_direct_module_api(self):
        self.replace(self.checker.JULIA, 'export measure\n"""', 'export measure\nif false\n"""')
        self.replace(
            self.checker.JULIA,
            "measure(value; scale=1) = value * scale\nend",
            "measure(value; scale=1) = value * scale\nend\nend",
        )
        with self.assertRaisesRegex(ValueError, "not directly in the Julia module body"):
            self.inventory()

    def test_r_source_comments_must_cover_real_parameters(self):
        self.replace(self.checker.R_SOURCE, "@param scale", "@param unrelated")
        with self.assertRaisesRegex(ValueError, "missing=\\['scale'\\].*unknown=\\['unrelated'\\]"):
            self.inventory()

    def test_r_source_errors_cannot_be_empty(self):
        self.replace(self.checker.R_SOURCE, "Arithmetic failures propagate from the number type.", "")
        with self.assertRaisesRegex(ValueError, "R source Errors section"):
            self.inventory()

    def test_r_alias_without_usage_is_not_complete_documentation(self):
        self.replace(self.manual, r"\usage{measure(value, scale = 1L)}", r"\usage{}")
        self.replace(self.manual, r"\item{value}{Input measurement in its original units.}", "")
        self.replace(self.manual, r"\item{scale}{Dimensionless multiplier defaulting to one.}", "")
        with self.assertRaisesRegex(ValueError, "missing R manual alias or usage"):
            self.inventory()

    def test_r_usage_default_drift_fails(self):
        self.replace(self.manual, "scale = 1L", "scale = 2L")
        with self.assertRaisesRegex(ValueError, "usage differs from source"):
            self.inventory()

    def test_r_manual_parameter_mentions_are_not_argument_entries(self):
        self.replace(self.manual, r"\item{scale}{Dimensionless multiplier defaulting to one.}", "")
        with self.assertRaisesRegex(ValueError, "missing=\\['scale'\\]"):
            self.inventory()

    def test_r_nested_defaults_preserve_parameter_boundaries(self):
        default = "scale = if (isTRUE(value)) 1L else 0L"
        self.replace(self.manual, "scale = 1L", default)
        self.replace(self.checker.R_SOURCE, "scale = 1L", default)
        self.assertEqual(self.inventory()["symbols"]["r"][0]["parameters"], ["value", "scale"])

    def test_public_rebinding_and_dynamic_namespace_fail(self):
        with (self.root / self.checker.R_SOURCE).open("a") as file:
            file.write("measure <- other_function\n")
        with self.assertRaisesRegex(ValueError, "one supported R source definition"):
            self.inventory()
        self.replace(self.checker.R_SOURCE, "measure <- other_function", "")
        self.replace("bindings/r/NAMESPACE", "export(measure)", 'exportPattern(".*")')
        with self.assertRaisesRegex(ValueError, "unsupported R namespace"):
            self.inventory()

    def test_nested_r_definition_does_not_satisfy_public_export(self):
        path = self.root / self.checker.R_SOURCE
        path.write_text("private <- function() {\n" + path.read_text() + "}\n")
        with self.assertRaisesRegex(ValueError, "nested inside an R expression"):
            self.inventory()

    def test_trailing_rd_comment_terminates_without_losing_sections(self):
        with (self.root / self.manual).open("a") as file:
            file.write("% A final comment without a trailing newline")
        self.assertTrue(self.inventory()["structural_complete"])

    def test_unknown_source_files_are_not_ignored(self):
        self.write("bindings/julia/src/new.jl", "export new_symbol\n")
        with self.assertRaisesRegex(ValueError, "source-file scope changed"):
            self.inventory()

    def test_unknown_workflow_command_cannot_be_recorded_as_supported(self):
        self.replace(self.checker.WORKFLOW, self.checker.COMMANDS["r_native_suite"], "Rscript different-test.R")
        with self.assertRaisesRegex(ValueError, "CI qualification command changed"):
            self.inventory()

    def test_quoted_default_spaces_remain_significant(self):
        self.assertNotEqual(self.checker._canonical('f(label="a b")'), self.checker._canonical('f(label="ab")'))

    def test_duplicate_manual_aliases_fail(self):
        self.write("bindings/r/man/duplicate.Rd", (self.root / self.manual).read_text())
        with self.assertRaisesRegex(ValueError, "duplicate R manual alias"):
            self.inventory()

    def test_missing_julia_example_is_not_classified_away(self):
        self.replace(self.checker.JULIA, "```julia\n@assert measure(2) == 2\n```", "")
        with self.assertRaisesRegex(ValueError, "Julia example membership changed"):
            self.inventory()

    def test_missing_r_example_is_not_classified_away(self):
        self.replace(self.manual, "\\examples{\nstopifnot(measure(2) == 2)\n}", "")
        with self.assertRaisesRegex(ValueError, "R example membership changed"):
            self.inventory()

    def test_julia_assertion_in_comment_does_not_count(self):
        self.replace(self.checker.JULIA, "@assert measure(2) == 2", "# @assert measure(2) == 2")
        with self.assertRaisesRegex(ValueError, "missing executable julia outcome assertion"):
            self.inventory()

    def test_r_assertion_in_string_does_not_count(self):
        self.replace(self.manual, "stopifnot(measure(2) == 2)", '"stopifnot(measure(2) == 2)"')
        with self.assertRaisesRegex(ValueError, "missing executable r outcome assertion"):
            self.inventory()

    def test_alternative_julia_fence_language_requires_review(self):
        self.replace(self.checker.JULIA, "```julia", "```jldoctest")
        with self.assertRaisesRegex(ValueError, "only literal unskipped"):
            self.inventory()

    def test_r_dontrun_wrapper_does_not_pass_as_executed_example(self):
        self.replace(self.manual, "stopifnot(measure(2) == 2)", r"\dontrun{stopifnot(measure(2) == 2)}")
        with self.assertRaisesRegex(ValueError, "skipped or conditional Rd example"):
            self.inventory()

    def test_missing_runner_fails_even_with_ci_command_present(self):
        (self.root / self.checker.RUNNERS["r"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing or outside-repository input.*check_r_documentation"):
            self.inventory()

    def test_missing_documentation_ci_command_fails(self):
        self.replace(self.checker.WORKFLOW, self.checker.DOCUMENTATION_COMMANDS["julia"], "")
        with self.assertRaisesRegex(ValueError, "CI qualification command changed: julia"):
            self.inventory()

    def test_commented_ci_command_is_not_configured_execution(self):
        command = self.checker.DOCUMENTATION_COMMANDS["r"]
        self.replace(self.checker.WORKFLOW, "- run: " + command, "# - run: " + command)
        with self.assertRaisesRegex(ValueError, "CI qualification command changed: r"):
            self.inventory()

    def test_literal_workflow_run_block_contains_documentation_command(self):
        command = self.checker.DOCUMENTATION_COMMANDS["julia"]
        self.replace(
            self.checker.WORKFLOW, "- run: " + command, "- run: |\n    # The command below runs in CI.\n    " + command
        )
        self.assertTrue(self.inventory()["structural_complete"])

    def test_runner_change_invalidates_stored_source_identity(self):
        self.record()
        self.write(self.checker.RUNNERS["julia"], "# A changed runner is not the recorded source.\n")
        self.assertTrue(self.checker.check_inventory(self.root))

    def test_example_capture_and_normalized_hashes_are_exact(self):
        inventory = self.inventory()
        julia = inventory["native_examples"]["julia"][0]
        r = inventory["native_examples"]["r"][0]
        self.assertEqual(julia["code"], "@assert measure(2) == 2\n")
        self.assertEqual(r["code"], "\nstopifnot(measure(2) == 2)\n")
        self.assertEqual(julia["assertion_count"], 1)
        self.assertEqual(r["assertion_count"], 1)
        self.record()
        for relative in (self.checker.JULIA, self.manual):
            path = self.root / relative
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(self.checker.check_inventory(self.root), [])


if __name__ == "__main__":
    unittest.main()
