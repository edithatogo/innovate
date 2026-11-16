"""H06 frozen hype impact benchmark protocol tests."""

from __future__ import annotations

import json
from pathlib import Path

from validation.hype.run import fit_and_score_families


def test_no_peak_data_is_not_forced_into_a_hype_cycle():
    protocol = json.loads(Path("validation/hype/protocol.json").read_text(encoding="utf-8"))
    time = [float(i) for i in range(15)]
    flat_observed = [2.0 for _ in time]
    result = fit_and_score_families(time, flat_observed, protocol)
    assert result["selected_family"] == "no_cycle"
    assert result["train_points"] + result["test_points"] == len(time)
