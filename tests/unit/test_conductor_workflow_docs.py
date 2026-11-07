"""Tests for Conductor workflow and tech-stack governance decisions."""

from __future__ import annotations

import re
from pathlib import Path


def test_kairos_runtime_acceptance_excludes_optional_game_prerequisites() -> None:
    """Runtime acceptance keeps its native gates without waiting for game profiles."""
    plan = Path("conductor/tracks/kairos_runtime_boundary_20261107/plan.md").read_text()
    sections = dict(
        re.findall(
            r"^## (I\d{2}[a-z]?):[^\n]*\n(.*?)(?=^## |\Z)",
            plan,
            re.MULTILINE | re.DOTALL,
        )
    )
    dependencies: dict[str, set[str]] = {}
    for identifier, body in sections.items():
        requires = re.search(r"^Executor:.*?Requires:([^\n]+)", body, re.MULTILINE)
        assert requires is not None, identifier
        declared = requires.group(1)
        dependencies[identifier] = set(re.findall(r"\bI\d{2}[a-z]?\b", declared))
        for first, last in re.findall(r"I(\d{2})[\u2013-]I(\d{2})", declared):
            dependencies[identifier].update(f"I{number:02}" for number in range(int(first), int(last) + 1))

    required: set[str] = set()
    pending = list(dependencies["I10"])
    while pending:
        identifier = pending.pop()
        assert identifier in dependencies, identifier
        if identifier not in required:
            required.add(identifier)
            pending.extend(dependencies[identifier])

    assert "I07" not in required
    assert {"I00", "I01", "I02", "I03", "I04", "I05", "I06", "I08", "I08b", "I09"} <= required
    assert "UOGTO executable profile" in sections["I07"]
    assert "generic Rust game package" in sections["I07"]
    assert dependencies["I07"] == set()
