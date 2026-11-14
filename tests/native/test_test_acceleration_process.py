"""Interrupted local feedback cannot leave an active writer or reusable stamp."""

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# Keep normal imports working for both pytest entrypoints without a global path change.
with pytest.MonkeyPatch.context() as import_path:
    import_path.syspath_prepend(str(REPO))
    profiles = importlib.import_module("scripts.test_acceleration.profiles")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups and SIGINT; Windows uses taskkill")
@pytest.mark.parametrize("mode", ["interrupt", "timeout"])
def test_process_tree_stops_before_cache_lease_is_released(tmp_path, mode):
    """Use an isolated supervisor so a real SIGINT cannot interrupt pytest."""
    supervisor = textwrap.dedent(
        """
        import json
        import os
        import signal
        import subprocess
        import sys
        import threading
        import time
        from pathlib import Path
        from scripts.test_acceleration import profiles

        directory = Path(sys.argv[1])
        mode = sys.argv[2]
        ready = directory / "ready.json"
        log = directory / "child.log"
        lease = directory / "active.lock"
        child_code = (
            "import json,os,subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "Path(sys.argv[1]).write_text(json.dumps({'parent':os.getpid(),'child':child.pid})); "
            "time.sleep(30)"
        )
        result = {"interrupted": False, "kill_observed": False}
        original_killpg = os.killpg

        def observed_killpg(group, sig):
            result["kill_observed"] = True
            result["lease_held_when_stopping"] = lease.is_dir()
            if ready.exists():
                pids = json.loads(ready.read_text())
                result["same_group"] = all(os.getpgid(pid) == group for pid in pids.values())
            return original_killpg(group, sig)

        profiles.os.killpg = observed_killpg

        def interrupt_when_ready():
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if ready.exists():
                    try:
                        json.loads(ready.read_text())
                    except ValueError:
                        continue
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                time.sleep(0.01)

        timer = None
        try:
            try:
                with profiles.cache_lease(directory):
                    if mode == "interrupt":
                        timer = threading.Thread(target=interrupt_when_ready, daemon=True)
                        timer.start()
                    code, _ = profiles.run_process(
                        [sys.executable, "-c", child_code, str(ready)],
                        directory, dict(os.environ), log, 2 if mode == "timeout" else 10,
                    )
                    result["exit_code"] = code
            except KeyboardInterrupt:
                result["interrupted"] = True
            if timer is not None:
                timer.join(timeout=1)
            pids = json.loads(ready.read_text())
            try:
                os.kill(pids["parent"], 0)
            except ProcessLookupError:
                result["parent_reaped"] = True
            else:
                result["parent_reaped"] = False
            # A killed orphan can briefly remain a zombie until init reaps it.
            # A zombie is no longer an active cache writer.
            deadline = time.monotonic() + 1
            while True:
                states = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pids["child"])],
                    capture_output=True, text=True, check=False,
                ).stdout.strip()
                result["descendant_stopped"] = not states or states.startswith("Z")
                if result["descendant_stopped"] or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            with profiles.cache_lease(directory):
                result["lease_reacquired_after_cleanup"] = result["parent_reaped"] and result["descendant_stopped"]
            result["lease_released"] = not lease.exists()
        finally:
            # Always clean only our own child group, including on a regression.
            if ready.exists():
                pids = json.loads(ready.read_text())
                try:
                    original_killpg(pids["parent"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print(json.dumps(result))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", supervisor, str(tmp_path), mode],
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["kill_observed"]
    assert observed["lease_held_when_stopping"]
    assert observed["same_group"]
    assert observed["parent_reaped"]
    assert observed["descendant_stopped"]
    assert observed["lease_reacquired_after_cleanup"]
    assert observed["lease_released"]
    assert observed["interrupted"] is (mode == "interrupt")
    if mode == "timeout":
        assert observed["exit_code"] == 124


@pytest.fixture
def seeded_cache(tmp_path, monkeypatch):
    """Prepare a synthetic prior success without requiring optional testmon."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\nminversion = '9.2'\n")
    inventory = {"python": "3.15.2", "pytest": "9.2.0"}
    monkeypatch.setattr(profiles, "ROOT", tmp_path)
    # These tests exercise the cache lifecycle in ordinary CI, not a selective
    # test run. Environment/profile opt-in guards are tested independently.
    monkeypatch.setattr(profiles, "validate_environment", lambda *args: None)
    monkeypatch.setattr(profiles, "_require_tools", lambda mode: {"pytest-testmon": "2.3.0"})
    monkeypatch.setattr(profiles, "runtime_inventory", lambda: inventory)
    monkeypatch.setattr(profiles, "native_binaries", lambda: [])
    cache = tmp_path / ".pytest_cache/test_acceleration"
    state_dir = cache / "testmon-native"
    state_dir.mkdir(parents=True)
    database = state_dir / ".testmondata"
    database.write_bytes(b"synthetic prior database")
    state = state_dir / "state.json"
    state.write_text(json.dumps({"snapshot": profiles.snapshot(tmp_path, "native", inventory), "inventory_count": 1}))
    return cache, database, state


@pytest.mark.parametrize("failure", ["test_failure", "timeout", "interrupt", "startup_error"])
def test_failed_warm_run_discards_stamp_and_forces_complete_recovery(seeded_cache, monkeypatch, failure):
    cache, database, state = seeded_cache

    def failed_run(command, directory, env, log, timeout):
        assert (cache / "active.lock").is_dir()
        assert not state.exists(), "The previous success must be invalid before a writer starts"
        assert "--testmon-noselect" not in command, "This starts as a warm run"
        if failure == "interrupt":
            raise KeyboardInterrupt("synthetic interrupted writer")
        if failure == "startup_error":
            raise OSError("synthetic process startup failure")
        code = 124 if failure == "timeout" else 1
        Path(env["INNOVATE_TEST_METRICS"]).write_text(
            json.dumps(
                {
                    "collected": 1,
                    "selected": 1,
                    "executed": 1,
                    "passed": 0,
                    "failed": 1,
                    "errors": 0,
                    "skipped": 0,
                    "exitstatus": code,
                }
            )
        )
        return code, 0.01

    monkeypatch.setattr(profiles, "run_process", failed_run)
    if failure in ("interrupt", "startup_error"):
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
            profiles.run_profile("testmon", "native", 30, False)
    else:
        receipt, code = profiles.run_profile("testmon", "native", 30, False)
        assert code == 1
        assert receipt["status"] == "failed"
    assert not state.exists()
    assert not (cache / "active.lock").exists()

    def recovered_run(command, directory, env, log, timeout):
        assert "--testmon-noselect" in command
        assert not state.exists()
        assert not database.exists(), "The suspect database must be discarded before recovery"
        database.write_bytes(b"synthetic complete rebuilt database")
        Path(env["INNOVATE_TEST_METRICS"]).write_text(
            json.dumps(
                {
                    "collected": 2,
                    "selected": 2,
                    "executed": 2,
                    "passed": 2,
                    "failed": 0,
                    "errors": 0,
                    "skipped": 0,
                    "exitstatus": 0,
                }
            )
        )
        return 0, 0.01

    monkeypatch.setattr(profiles, "run_process", recovered_run)
    receipt, code = profiles.run_profile("testmon", "native", 30, False)
    assert code == 0
    assert receipt["status"] == "passed"
    assert "unseeded" in receipt["invalidation_reasons"]
    assert json.loads(state.read_text())["inventory_count"] == 2
