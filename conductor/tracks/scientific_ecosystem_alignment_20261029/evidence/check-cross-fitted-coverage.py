"""Check native cross-fitted ATE and ATT on a stratified synthetic design."""

import argparse
from random import Random
from statistics import mean

from innovate import execute

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clustered", action="store_true")
parser.add_argument("--few-clusters", action="store_true", help="Compare normal and t(9) intervals")
args = parser.parse_args()
clustered = args.clustered or args.few_clusters
cluster_count = 10 if args.few_clusters else 80
rng = Random(912)
strata = ["low", "high"] * 400
folds = [index // 2 % 5 for index in range(len(strata))]
if clustered:
    cluster_ids = [index // (len(strata) // cluster_count) for index in range(len(strata))]
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
covered_t = dict.fromkeys(methods, 0)
for _ in range(400):
    treatment = [int(rng.random() < p) for p in probabilities]
    if clustered:
        shocks = [rng.gauss(0, 2) for _ in range(cluster_count)]
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
        distance = abs(estimate - truth)
        error = result["influence_standard_error"]
        covered[key] += distance <= 1.96 * error
        if args.few_clusters:
            covered_t[key] += distance <= 2.262157 * error

for key, truth in methods.items():
    average = mean(estimates[key])
    print(f"{key}: mean {average:.6f}, 95% coverage {covered[key]}/400")
    assert abs(average - truth) < 0.04
    if args.few_clusters:
        print(f"{key}: t(9) coverage {covered_t[key]}/400")
    else:
        assert 360 <= covered[key] <= 395
