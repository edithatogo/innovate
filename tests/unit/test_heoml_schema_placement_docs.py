"""Governance checks for the HEOML schema placement decision."""

from __future__ import annotations

from pathlib import Path

ADR_PATH = Path("docs/adr/0005-heoml-schema-placement.md")
SCHEMA_HOME = Path("specs/ecosystem/heoml/extensions/innovate/README.md")


def test_heoml_schema_placement_decision_artifacts_exist() -> None:
    """The placement decision should be documented in ADR and schema specs."""
    assert ADR_PATH.is_file()
    assert SCHEMA_HOME.is_file()


def test_heoml_schema_placement_compares_all_options() -> None:
    """The ADR should compare the candidate schema homes and their tradeoffs."""
    decision = ADR_PATH.read_text()

    for option in (
        "repo-local `innovate` schemas",
        "embedded `lifecourse` schemas",
        "future standalone `heoml` repository",
        "ownership",
        "versioning",
        "compatibility",
        "publication",
        "migration",
    ):
        assert option in decision


def test_heoml_schema_placement_records_interim_home_and_migration_trigger() -> None:
    """The selected home and standalone migration trigger should be explicit."""
    decision = ADR_PATH.read_text()
    schema_home = SCHEMA_HOME.read_text()
    ecosystem_contract = Path("specs/ecosystem/README.md").read_text()
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()

    for document in (decision, schema_home, ecosystem_contract, strategy):
        assert "specs/ecosystem/heoml/extensions/innovate/" in document
        assert "standalone `heoml` repository" in document
        assert "migration trigger" in document
        assert "deprecation window" in document


def test_heoml_schema_contracts_remain_binding_friendly() -> None:
    """HEOML placement must preserve JSON and Arrow-compatible contracts."""
    combined = "\n".join(
        path.read_text()
        for path in (
            ADR_PATH,
            SCHEMA_HOME,
            Path("specs/ecosystem/README.md"),
            Path("docs/ecosystem/module_incubation_strategy.md"),
        )
    )

    for token in (
        "binding-friendly JSON",
        "Arrow-compatible",
        "JSON Schema",
        "schema_version",
        "semver",
        "MUST NOT use private Python objects",
        "MUST NOT use pickle",
        "private Python object framing",
    ):
        assert token in combined


def test_heoml_schema_placement_adr_is_indexed() -> None:
    """ADR navigation should include the HEOML placement decision."""
    index = Path("docs/adr/index.md").read_text()

    assert "0005-heoml-schema-placement" in index
