"""Check a bounded four-chain sampler against an independent Gaussian posterior."""

import json
from math import isfinite
from statistics import mean, variance

from innovate import execute


def main() -> None:
    """Check support, draws, diagnostics and a predeclared Monte Carlo tolerance."""
    result = execute(
        "bayesian_metropolis_chains",
        forecast_model="logistic",
        parameter="L",
        bounds=[80, 120],
        proposal_step=4,
        gaussian_priors={"L": {"mean": 96, "sd": 2}},
        noise_sd=1,
        observed=[50],
        prediction={"inputs": {"time": [2]}, "parameters": {"k": 0.5, "x0": 2}},
        draws=4000,
        warmup=1000,
        chains=[
            {"seed": 41, "initial": {"L": 83}},
            {"seed": 42, "initial": {"L": 94}},
            {"seed": 43, "initial": {"L": 106}},
            {"seed": 44, "initial": {"L": 118}},
        ],
    )["result"]
    chains = result["chains"]
    assert len(chains) == 4
    assert [chain["seed"] for chain in chains] == [41, 42, 43, 44]
    for chain in chains:
        assert chain["draws"] == 4000 and chain["warmup"] == 1000
        assert len(chain["samples"]) == 4000
        assert 0 < chain["acceptance_rate"] < 1
    samples = [sample for chain in chains for sample in chain["samples"]]
    assert all(isfinite(sample) and 80 <= sample <= 120 for sample in samples)

    # At the midpoint, y=L/2. Combining N(96, 4) with y=50, noise variance 1,
    # gives N(98, 2), truncated to [80, 120]. Both bounds exceed 12 SD from 98;
    # truncation is negligible at the declared tolerances, not mathematically absent.
    # These fixed tolerances check a seeded fixture; they are not interval coverage.
    assert abs(mean(samples) - 98.0) < 0.2
    assert abs(variance(samples) - 2.0) < 0.3
    assert len(result["diagnostics"]) == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["parameter"] == "L"
    assert diagnostic["effective_sample_size"]["draws_used"] == 16000
    rhat = diagnostic["split_rhat"]["rhat"]
    assert isfinite(rhat) and 0 < rhat < 1.05
    for name in ("bulk_ess", "tail_ess"):
        effective_samples = diagnostic["effective_sample_size"][name]
        assert isfinite(effective_samples) and effective_samples > 1000
    # Diagnostic thresholds are warning criteria, not proof of convergence.
    print(json.dumps({"chains": len(chains), "draws": len(samples), "posterior_oracle_passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
