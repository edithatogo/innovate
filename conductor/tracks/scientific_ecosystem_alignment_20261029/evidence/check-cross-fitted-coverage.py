"""Check native cross-fitted ATE and ATT on a stratified synthetic design."""

import argparse
from random import Random
from statistics import mean

from innovate import NativeError, execute

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clustered", action="store_true")
parser.add_argument("--few-clusters", action="store_true", help="Compare normal and t(9) intervals")
parser.add_argument("--weak-overlap", action="store_true")
parser.add_argument("--minimum-training-arm-count", type=int, default=1)
parser.add_argument("--cluster-assignment", action="store_true")
args = parser.parse_args()
if args.cluster_assignment and (args.few_clusters or args.weak_overlap):
    parser.error("cluster assignment uses the 80-cluster design with good overlap")
if args.minimum_training_arm_count < 1:
    parser.error("minimum training arm count must be positive")
clustered = args.clustered or args.few_clusters or args.cluster_assignment
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
low_probability = 0.01 if args.weak_overlap else 0.2
if args.cluster_assignment:
    strata = ["high" if cluster % 2 else "low" for cluster in cluster_ids]
probabilities = [low_probability if stratum == "low" else 1 - low_probability for stratum in strata]
baseline = [0.0 if stratum == "low" else 4.0 for stratum in strata]
effects = [1.0 if stratum == "low" else 3.0 for stratum in strata]
methods = {"ate": 2.0, "att": 3 - 2 * low_probability}
estimates = {key: [] for key in methods}
covered = dict.fromkeys(methods, 0)
failed = dict.fromkeys(methods, 0)
guarded = dict.fromkeys(methods, 0)
covered_t = dict.fromkeys(methods, 0)
for _ in range(400):
    if args.cluster_assignment:
        assigned = [int(rng.random() < p) for p in probabilities[:: len(strata) // cluster_count]]
        treatment = [assigned[cluster] for cluster in cluster_ids]
    else:
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
        try:
            result = execute(
                "summarize_causal",
                method=f"cross_fitted_stratum_{key}",
                outcomes=outcomes,
                treatment=treatment,
                strata=strata,
                folds=folds,
                minimum_training_arm_count=args.minimum_training_arm_count,
                **cluster_options,
            )["result"]["contrast"]
        except NativeError as error:
            if (args.weak_overlap or args.cluster_assignment) and "requires both arms" in str(error):
                failed[key] += 1
            elif args.weak_overlap and "falls below minimum_training_arm_count" in str(error):
                guarded[key] += 1
            else:
                raise
            continue
        assert result["cluster_degrees_of_freedom"] == (cluster_count - 1 if clustered else None)
        estimate = result["estimate"]
        estimates[key].append(estimate)
        distance = abs(estimate - truth)
        error = result["influence_standard_error"]
        covered[key] += distance <= 1.96 * error
        if args.few_clusters:
            covered_t[key] += distance <= 2.262157 * error

for key, truth in methods.items():
    average = mean(estimates[key])
    successful = len(estimates[key])
    print(f"{key}: mean {average:.6f}, 95% coverage {covered[key]}/{successful}, "
          f"missing-arm failures {failed[key]}/400, guard rejections {guarded[key]}/400")
    if args.weak_overlap:
        assert 0 < failed[key] < 400
        continue
    if args.cluster_assignment:
        assert successful + failed[key] + guarded[key] == 400
        continue
    assert abs(average - truth) < 0.04
    if args.few_clusters:
        print(f"{key}: t(9) coverage {covered_t[key]}/400")
    else:
        assert 360 <= covered[key] <= 395
