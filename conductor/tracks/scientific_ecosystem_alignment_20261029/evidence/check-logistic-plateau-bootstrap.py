"""Measure accepted-refit coverage with a sparse Logistic transition."""

from math import exp
from random import Random
from statistics import mean

from innovate import execute

rng = Random(272)
time = [-3, -2, -1, 0, 100, 101, 102, 103]
truth = {"L": 100, "k": 1, "x0": 0}
trials, draws = 40, 32
covered = dict.fromkeys(truth, 0)
available = dict.fromkeys(truth, 0)
successful_refits = []
rank_failures = 0
for trial in range(trials):
    observed = [100 / (1 + exp(-t)) + rng.gauss(0, 1) for t in time]
    result = execute(
        "bootstrap_fit",
        fit_model="logistic",
        draws=draws,
        seed=trial,
        fit={
            "inputs": {"time": time, "observed": observed},
            "fit_options": {"initial_parameters": truth},
        },
    )["result"]
    succeeded = result["succeeded"]
    assert succeeded + sum(result["failure_counts"].values()) == draws
    successful_refits.append(succeeded)
    rank_failures += result["failure_counts"]["rank_deficient"]
    for name, target in truth.items():
        interval = result["parameter_intervals"][name]
        if interval is not None:
            available[name] += 1
            covered[name] += interval[0] <= target <= interval[1]

print(f"Mean accepted refits: {mean(successful_refits):.2f}/{draws}")
print(f"Rank-deficient refits: {rank_failures}/{trials * draws}")
for name in truth:
    print(f"{name}: covered {covered[name]}/{available[name]} available intervals")
assert rank_failures > 0
