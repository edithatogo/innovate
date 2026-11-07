"""Check native cross-fitted ATE and ATT on a stratified synthetic design."""

from random import Random
from statistics import mean

from innovate import execute

rng = Random(912)
strata = ["low", "high"] * 400
folds = [index // 2 % 5 for index in range(len(strata))]
probabilities = [0.2, 0.8] * 400
baseline = [0.0, 4.0] * 400
effects = [1.0, 3.0] * 400
methods = {"ate": 2.0, "att": 2.6}
estimates = {key: [] for key in methods}
covered = dict.fromkeys(methods, 0)
for _ in range(400):
    treatment = [int(rng.random() < p) for p in probabilities]
    outcomes = [
        base + effect * flag + rng.gauss(0, 1)
        for base, effect, flag in zip(baseline, effects, treatment, strict=True)
    ]
    for key, truth in methods.items():
        result = execute(
            "summarize_causal",
            method=f"cross_fitted_stratum_{key}",
            outcomes=outcomes,
            treatment=treatment,
            strata=strata,
            folds=folds,
        )["result"]["contrast"]
        estimate = result["estimate"]
        estimates[key].append(estimate)
        covered[key] += abs(estimate - truth) <= 1.96 * result["influence_standard_error"]

for key, truth in methods.items():
    average = mean(estimates[key])
    print(f"{key}: mean {average:.6f}, 95% coverage {covered[key]}/400")
    assert abs(average - truth) < 0.04
    assert 360 <= covered[key] <= 395
