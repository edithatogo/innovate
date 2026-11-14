"""Count actual pytest activity without retaining test identities or payloads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    "collected",
    "selected",
    "deselected",
    "started",
    "executed",
    "passed",
    "failed",
    "skipped",
    "errors",
    "exitstatus",
    "duration_seconds",
}


def run_pytest(tmp_path, source, *args, conftest=None, enabled=True):
    """Execute a fresh isolated pytest process with the metrics plugin."""
    (tmp_path / "test_sample.py").write_text(source)
    (tmp_path / "pytest.ini").write_text("[pytest]\nminversion = 9.2\naddopts =\n")
    if conftest is not None:
        (tmp_path / "conftest.py").write_text(conftest)
    metrics_path = tmp_path / "metrics.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
    }
    env.pop("PYTEST_PLUGINS", None)
    env.pop("INNOVATE_TEST_METRICS", None)
    if enabled:
        env["INNOVATE_TEST_METRICS"] = str(metrics_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.test_acceleration.metrics",
            "--color=no",
            "-q",
            *args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if not enabled:
        assert not metrics_path.exists()
        return result, None
    assert metrics_path.is_file(), result.stdout + result.stderr
    metrics = json.loads(metrics_path.read_text())
    assert set(metrics) == FIELDS
    assert metrics["exitstatus"] == result.returncode
    assert metrics["duration_seconds"] >= 0
    for field in FIELDS - {"duration_seconds"}:
        assert type(metrics[field]) is int
    assert metrics["collected"] == metrics["selected"] + metrics["deselected"]
    return result, metrics


@pytest.mark.parametrize(
    ("source", "status", "expected"),
    [
        (
            "def test_pass():\n    assert True\n",
            0,
            {"started": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 0},
        ),
        (
            "def test_fail():\n    print('private test payload')\n    assert False, 'private failure payload'\n",
            1,
            {"started": 1, "executed": 1, "passed": 0, "failed": 1, "skipped": 0, "errors": 0},
        ),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError('private setup payload')\n"
            "def test_setup_error(broken):\n    pass\n",
            1,
            {"started": 1, "executed": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 1},
        ),
        (
            "import pytest\n@pytest.mark.skip(reason='private skip reason')\ndef test_skip():\n    pass\n",
            0,
            {"started": 1, "executed": 0, "passed": 0, "failed": 0, "skipped": 1, "errors": 0},
        ),
        (
            "import pytest\ndef test_skip_in_body():\n    pytest.skip('private skip reason')\n",
            0,
            {"started": 1, "executed": 1, "passed": 0, "failed": 0, "skipped": 1, "errors": 0},
        ),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    yield\n    raise RuntimeError('private teardown')\n"
            "def test_teardown_error(broken):\n    assert True\n",
            1,
            {"started": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 1},
        ),
    ],
    ids=["pass", "fail", "setup-error", "setup-skip", "call-skip", "teardown-error"],
)
def test_metrics_distinguish_execution_outcomes(tmp_path, source, status, expected):
    result, metrics = run_pytest(tmp_path, source)
    assert result.returncode == status, result.stdout + result.stderr
    assert metrics["collected"] == metrics["selected"] == 1
    assert metrics["deselected"] == 0
    assert {key: metrics[key] for key in expected} == expected
    receipt = (tmp_path / "metrics.json").read_text()
    assert "private" not in receipt
    assert "test_sample" not in receipt
    assert str(tmp_path) not in receipt


@pytest.mark.parametrize("order", ["tryfirst", "trylast"])
def test_metrics_count_deselection_independently_of_hook_order(tmp_path, order):
    source = "def test_keep():\n    pass\ndef test_remove():\n    pass\n"
    conftest = (
        "import pytest\n"
        f"@pytest.hookimpl({order}=True)\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    removed = [item for item in items if item.name == 'test_remove']\n"
        "    items[:] = [item for item in items if item.name != 'test_remove']\n"
        "    config.hook.pytest_deselected(items=removed)\n"
    )
    result, metrics = run_pytest(tmp_path, source, conftest=conftest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metrics["collected"] == 2
    assert metrics["selected"] == metrics["deselected"] == 1
    assert metrics["started"] == metrics["executed"] == metrics["passed"] == 1


@pytest.mark.parametrize(
    ("source", "args", "collected", "deselected"),
    [
        ("", [], 0, 0),
        ("def test_excluded():\n    pass\n", ["-k", "never_matches"], 1, 1),
    ],
    ids=["empty-inventory", "all-deselected"],
)
def test_metrics_preserve_empty_collection_and_selection(tmp_path, source, args, collected, deselected):
    result, metrics = run_pytest(tmp_path, source, *args)
    assert result.returncode == 5, result.stdout + result.stderr
    assert metrics["collected"] == collected
    assert metrics["deselected"] == deselected
    assert metrics["selected"] == metrics["started"] == metrics["executed"] == 0


def test_metrics_record_collection_failure(tmp_path):
    result, metrics = run_pytest(tmp_path, "raise RuntimeError('private collection payload')\n")
    assert result.returncode == 2, result.stdout + result.stderr
    assert metrics["collected"] == metrics["selected"] == metrics["started"] == metrics["executed"] == 0
    assert metrics["failed"] == 0
    assert metrics["errors"] == 1


def test_metrics_do_not_invent_tests_for_collection_skip(tmp_path):
    result, metrics = run_pytest(
        tmp_path, "import pytest\npytest.skip('private module skip', allow_module_level=True)\n"
    )
    assert result.returncode == 5, result.stdout + result.stderr
    assert metrics["collected"] == metrics["selected"] == metrics["started"] == metrics["executed"] == 0
    assert metrics["skipped"] == 1
    assert metrics["errors"] == 0


def test_metrics_count_each_call_once(tmp_path):
    conftest = (
        "def pytest_configure(config):\n"
        "    global configuration\n"
        "    configuration = config\n"
        "def pytest_runtest_logreport(report):\n"
        "    configuration.pluginmanager.get_plugin('innovate-test-metrics').pytest_runtest_logreport(report)\n"
    )
    result, metrics = run_pytest(tmp_path, "def test_pass():\n    pass\n", conftest=conftest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metrics["executed"] == metrics["passed"] == 1


def test_metrics_plugin_is_inactive_without_opt_in(tmp_path):
    result, metrics = run_pytest(tmp_path, "def test_pass():\n    pass\n", enabled=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metrics is None
