"""Unit and protocol tests for the I04 SBC and coverage calibration harness."""

from __future__ import annotations

import json
from pathlib import Path

from validation.inference.run import conjugate_normal_posterior_sampler, evaluate_protocol


def _load_protocol() -> dict:
    return json.loads(Path("validation/inference/protocol.json").read_text(encoding="utf-8"))


def test_exact_conjugate_normal_calibrates_within_monte_carlo_bounds():
    protocol = _load_protocol()
    result = evaluate_protocol(protocol)
    assert result["failed_replicates"] == 0
    assert result["calibrated"] is True
    assert abs(result["empirical_coverage"] - protocol["nominal_coverage"]) <= protocol["coverage_tolerance"]
    assert abs(result["mean_normalized_rank"] - 0.5) <= protocol["mean_rank_tolerance"]


def test_injected_bias_and_fake_narrow_intervals_are_detected():
    protocol = _load_protocol()
    cfg = protocol["conjugate_normal"]

    biased = evaluate_protocol(
        protocol,
        sampler=lambda obs, r: conjugate_normal_posterior_sampler(
            obs,
            prior_mean=cfg["prior_mean"],
            prior_sd=cfg["prior_sd"],
            obs_sd=cfg["obs_sd"],
            n_draws=cfg["n_draws"],
            rng=r,
            bias=1.5,
        ),
    )
    assert biased["calibrated"] is False
    assert biased["empirical_coverage"] < protocol["nominal_coverage"] - protocol["coverage_tolerance"]

    narrow = evaluate_protocol(
        protocol,
        sampler=lambda obs, r: conjugate_normal_posterior_sampler(
            obs,
            prior_mean=cfg["prior_mean"],
            prior_sd=cfg["prior_sd"],
            obs_sd=cfg["obs_sd"],
            n_draws=cfg["n_draws"],
            rng=r,
            interval_scale=0.2,
        ),
    )
    assert narrow["calibrated"] is False
    assert narrow["empirical_coverage"] < 0.6


def test_failed_replicates_count_in_denominator():
    protocol = _load_protocol()
    cfg = protocol["conjugate_normal"]
    counter = {"n": 0}

    def flaky_sampler(obs, r):
        counter["n"] += 1
        if counter["n"] % 4 == 0:
            return None
        return conjugate_normal_posterior_sampler(
            obs,
            prior_mean=cfg["prior_mean"],
            prior_sd=cfg["prior_sd"],
            obs_sd=cfg["obs_sd"],
            n_draws=cfg["n_draws"],
            rng=r,
        )

    result = evaluate_protocol(protocol, sampler=flaky_sampler)
    assert result["failed_replicates"] == protocol["replicates"] // 4
    assert result["failure_rate"] == 0.25
    # Even if every valid replicate were covered, 25% failures caps empirical_coverage at <= 0.75.
    assert result["empirical_coverage"] <= 0.75
    assert result["calibrated"] is False
