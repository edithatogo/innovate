"""Check native cross-fitted ATE and ATT on a stratified synthetic design."""

import argparse
from random import Random
from statistics import mean

from innovate import execute

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clustered", action="store_true")
args = parser.parse_args()
rng = Random(912)
strata = ["low", "high"] * 400
folds = [index // 2 % 5 for index in range(len(strata))]
if args.clustered:
    cluster_ids = [index // 10 for index in range(len(strata))]
    folds = [cluster % 5 for cluster in cluster_ids]
    cluster_options = {"clusters": [str(cluster) for cluster in cluster_ids]}
else:
    cluster_options = {}
probabilities = [0.2, 0.8] * 400
baseline = [0.0, 4.0] * 400
effects = [1.0, 3.0] * 400
methods = {"ate": 2.0, "att": 2.6}
estimates = {key: [] for key in methods}
covered = dict.fromkeys(methods, 0)
for _ in range(400):
    treatment = [int(rng.random() < p) for p in probabilities]
    if args.clustered:
        shocks = [rng.gauss(0, 2) for _ in range(80)]
        offsets = [shocks[cluster] for cluster in cluster_ids]
    else:
        offsets = [0.0] * len(strata)
    outcomes = [
        base + effect * flag + rng.gauss(0, 1) + offset
        for base, effect, flag, offset in zip(baseline, effects, treatment, offsets, strict=True)
    ]
    for key, truth in methods.items():
        result = execute(
            "summarize_causal",
            method=f"cross_fitted_stratum_{key}",
            outcomes=outcomes,
            treatment=treatment,
            strata=strata,
            folds=folds,
            **cluster_options,
        )["result"]["contrast"]
        estimate = result["estimate"]
        estimates[key].append(estimate)
        covered[key] += abs(estimate - truth) <= 1.96 * result["influence_standard_error"]

for key, truth in methods.items():
    average = mean(estimates[key])
    print(f"{key}: mean {average:.6f}, 95% coverage {covered[key]}/400")
    assert abs(average - truth) < 0.04
    assert 360 <= covered[key] <= 395
