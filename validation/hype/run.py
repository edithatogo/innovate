"""H06 frozen hype impact benchmark protocol runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def fit_and_score_families(
    time: list[float],
    observed: list[float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Compare no_cycle vs ordered_cycle using strictly training-only split and BIC."""
    n = len(time)
    if n < 6 or len(observed) != n:
        raise ValueError("aligned series with >= 6 points required")
    split = max(3, min(n - 1, math.floor(n * float(protocol["train_fraction"]))))
    train_y = observed[:split]
    test_y = observed[split:]

    # No-cycle baseline: linear/constant mean on training data only (k=1 param).
    mean_train = sum(train_y) / len(train_y)
    train_sse_no = sum((y - mean_train) ** 2 for y in train_y)
    test_rmse_no = math.sqrt(sum((y - mean_train) ** 2 for y in test_y) / len(test_y))
    bic_no = len(train_y) * math.log(max(train_sse_no / len(train_y), 1e-12)) + math.log(len(train_y))

    # Peak-pulse comparator (k=4 params): Gaussian pulse centered at training peak.
    peak_idx = max(range(len(train_y)), key=lambda i: train_y[i])
    peak_val = train_y[peak_idx]
    peak_t = time[peak_idx]
    width = max(1.0, (time[split - 1] - time[0]) / 4.0)
    amp = peak_val - mean_train
    train_sse_cycle = sum(
        (y - (mean_train + amp * math.exp(-0.5 * ((t - peak_t) / width) ** 2))) ** 2
        for t, y in zip(time[:split], train_y, strict=True)
    )
    bic_cycle = len(train_y) * math.log(max(train_sse_cycle / len(train_y), 1e-12)) + 4 * math.log(len(train_y))
    margin = float(protocol["no_peak_bic_margin"])
    selected = "ordered_cycle" if bic_cycle + margin < bic_no and (peak_val - mean_train) > 0.5 else "no_cycle"

    return {
        "train_points": split,
        "test_points": len(test_y),
        "selected_family": selected,
        "test_rmse_no_cycle": test_rmse_no,
        "bic_no_cycle": bic_no,
        "bic_ordered_cycle": bic_cycle,
    }


def main() -> int:
    protocol = json.loads(Path("validation/hype/protocol.json").read_text(encoding="utf-8"))
    time = [float(i) for i in range(12)]
    observed = [1.0 for _ in time]
    summary = fit_and_score_families(time, observed, protocol)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
