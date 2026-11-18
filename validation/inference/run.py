"""Simulation-based calibration (SBC) and repeated-coverage harness (I04)."""

# Temporary source change for the isolated CodeQL pull-request probe.

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("cannot compute quantile of empty draws")
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def conjugate_normal_posterior_sampler(
    observations: list[float],
    *,
    prior_mean: float,
    prior_sd: float,
    obs_sd: float,
    n_draws: int,
    rng: random.Random,
    bias: float = 0.0,
    interval_scale: float = 1.0,
) -> list[float]:
    """Exact conjugate Gaussian posterior draws for Normal(mu, obs_sd^2) with Normal prior."""
    n = len(observations)
    prior_prec = 1.0 / (prior_sd * prior_sd)
    lik_prec = n / (obs_sd * obs_sd)
    post_prec = prior_prec + lik_prec
    post_var = 1.0 / post_prec
    post_sd = math.sqrt(post_var) * interval_scale
    post_mean = (prior_prec * prior_mean + (sum(observations) / (obs_sd * obs_sd))) * post_var + bias
    return [rng.gauss(post_mean, post_sd) for _ in range(n_draws)]


def evaluate_protocol(
    protocol: dict[str, Any],
    sampler: Callable[..., list[float] | None] | None = None,
) -> dict[str, Any]:
    """Run SBC + repeated coverage over `protocol['replicates']` replicates.

    Failed replicates (`sampler(...) is None` or raises) are counted in the
    denominator of both `empirical_coverage` and `failure_rate`.
    """
    replicates = int(protocol["replicates"])
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    seed = int(protocol["seed"])
    nominal = float(protocol["nominal_coverage"])
    cov_tol = float(protocol["coverage_tolerance"])
    rank_tol = float(protocol["mean_rank_tolerance"])
    max_fail = float(protocol["max_failure_rate"])
    cfg = protocol["conjugate_normal"]
    prior_mean = float(cfg["prior_mean"])
    prior_sd = float(cfg["prior_sd"])
    obs_sd = float(cfg["obs_sd"])
    n_obs = int(cfg["n_obs"])
    n_draws = int(cfg["n_draws"])

    rng = random.Random(seed)
    tail = (1.0 - nominal) / 2.0
    if sampler is None:

        def _default_sampler(obs: list[float], r: random.Random) -> list[float]:
            return conjugate_normal_posterior_sampler(
                obs,
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                obs_sd=obs_sd,
                n_draws=n_draws,
                rng=r,
            )

        sampler = _default_sampler

    records: list[dict[str, Any]] = []
    covered_count = 0
    failed_count = 0
    norm_ranks: list[float] = []

    for idx in range(replicates):
        theta_true = rng.gauss(prior_mean, prior_sd)
        observations = [rng.gauss(theta_true, obs_sd) for _ in range(n_obs)]
        draws: list[float] | None
        try:
            draws = sampler(observations, rng)
        except Exception:
            draws = None

        if not draws or any(not math.isfinite(x) for x in draws):
            failed_count += 1
            records.append(
                {
                    "replicate": idx,
                    "status": "invalid",
                    "theta_true": theta_true,
                    "covered": False,
                    "sbc_rank": None,
                }
            )
            continue

        sorted_draws = sorted(draws)
        lo = _quantile(sorted_draws, tail)
        hi = _quantile(sorted_draws, 1.0 - tail)
        covered = lo <= theta_true <= hi
        if covered:
            covered_count += 1
        rank = sum(1 for d in sorted_draws if d < theta_true)
        norm_ranks.append(rank / len(sorted_draws))
        records.append(
            {
                "replicate": idx,
                "status": "valid",
                "theta_true": theta_true,
                "interval": [lo, hi],
                "covered": covered,
                "sbc_rank": rank,
            }
        )

    failure_rate = failed_count / replicates
    empirical_coverage = covered_count / replicates
    mean_norm_rank = sum(norm_ranks) / len(norm_ranks) if norm_ranks else float("nan")
    calibrated = (
        failure_rate <= max_fail
        and abs(empirical_coverage - nominal) <= cov_tol
        and math.isfinite(mean_norm_rank)
        and abs(mean_norm_rank - 0.5) <= rank_tol
    )

    return {
        "schema_version": protocol["schema_version"],
        "total_replicates": replicates,
        "valid_replicates": replicates - failed_count,
        "failed_replicates": failed_count,
        "failure_rate": failure_rate,
        "nominal_coverage": nominal,
        "empirical_coverage": empirical_coverage,
        "mean_normalized_rank": mean_norm_rank,
        "calibrated": calibrated,
        "replicates": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SBC and coverage protocol.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("validation/inference/protocol.json"),
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary = evaluate_protocol(protocol)
    printable = {k: v for k, v in summary.items() if k != "replicates"}
    print(json.dumps(printable, indent=2))
    return 0 if summary["calibrated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
