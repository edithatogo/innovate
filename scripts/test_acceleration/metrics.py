"""Opt-in pytest counts for local feedback measurements.

Set ``INNOVATE_TEST_METRICS`` to an output filename and load this plugin with
``-p scripts.test_acceleration.metrics``. Receipts contain numbers only: no test
identities, captured output, failure messages or local paths. ``executed`` counts
unique call reports, including calls that skip; a setup skip has no call report.
``skipped`` reports those skips separately, including collection skips. ``errors``
counts collection and setup/teardown failures separately from failed test calls.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest


class _Metrics:
    def __init__(self, output: Path):
        self.output = output
        self.start = time.perf_counter()
        self.selected = 0
        self.deselected = 0
        self.started: set[str] = set()
        self.executed: set[str] = set()
        self.passed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()
        self.errors: set[tuple[str, str]] = set()

    def pytest_sessionstart(self) -> None:
        self.start = time.perf_counter()

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected += len(items)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        # All collection_modifyitems hooks have finished here. testscollected is
        # assigned afterwards by pytest, so inspect the final item list directly.
        self.selected = len(session.items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.errors.add(("collect", report.nodeid))
        elif report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        self.started.add(nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call":
            self.executed.add(report.nodeid)
            if report.passed:
                self.passed.add(report.nodeid)
            elif report.failed:
                self.failed.add(report.nodeid)
        elif report.failed:
            self.errors.add((report.when, report.nodeid))
        if report.skipped:
            self.skipped.add(report.nodeid)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, exitstatus: int) -> None:
        receipt = {
            "collected": self.selected + self.deselected,
            "selected": self.selected,
            "deselected": self.deselected,
            "started": len(self.started),
            "executed": len(self.executed),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "exitstatus": int(exitstatus),
            "duration_seconds": time.perf_counter() - self.start,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
        try:
            temporary.replace(self.output)
        finally:
            temporary.unlink(missing_ok=True)


def pytest_configure(config: pytest.Config) -> None:
    """Register measurement hooks only when the caller explicitly opts in."""
    output = os.environ.get("INNOVATE_TEST_METRICS")
    if output:
        config.pluginmanager.register(_Metrics(Path(output)), "innovate-test-metrics")
