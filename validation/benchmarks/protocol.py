"""V02 leakage-safe temporal benchmark protocol."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Frozen temporal benchmark specification preventing future leakage."""

    vintage: str
    origin_index: int
    horizon: int
    seed: int
    series: tuple[float, ...]

    def validate_split(self, train_indices: list[int], holdout_indices: list[int]) -> None:
        if not train_indices or not holdout_indices:
            raise ValueError("train and holdout splits must be nonempty")
        if max(train_indices) >= self.origin_index:
            raise ValueError("training indices leak beyond origin_index")
        if min(holdout_indices) < self.origin_index:
            raise ValueError("holdout indices overlap training window")
        if len(set(holdout_indices)) != len(holdout_indices):
            raise ValueError("duplicate holdout indices detected")
        if len(holdout_indices) > self.horizon:
            raise ValueError("holdout exceeds declared horizon")

    def run(
        self,
        predictor: Callable[[list[float], int, int], list[float] | None],
    ) -> dict[str, object]:
        train_idx = list(range(self.origin_index))
        hold_idx = list(range(self.origin_index, min(len(self.series), self.origin_index + self.horizon)))
        self.validate_split(train_idx, hold_idx)

        # Training scaling uses strictly training observations (never full-series max/mean).
        train_y = [self.series[i] for i in train_idx]
        hold_y = [self.series[i] for i in hold_idx]
        t0 = time.perf_counter()
        try:
            preds = predictor(train_y, len(hold_y), self.seed)
        except (ValueError, RuntimeError, TypeError):
            preds = None
        elapsed = time.perf_counter() - t0

        if preds is None or len(preds) != len(hold_y):
            return {
                "vintage": self.vintage,
                "seed": self.seed,
                "status": "failed",
                "rmse": None,
                "elapsed_s": elapsed,
            }
        mse = sum((p - y) ** 2 for p, y in zip(preds, hold_y, strict=True)) / len(hold_y)
        return {
            "vintage": self.vintage,
            "seed": self.seed,
            "status": "ok",
            "rmse": mse ** 0.5,
            "elapsed_s": elapsed,
        }
