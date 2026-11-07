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
parser.add_argument("--minimum-training-arm-clusters", type=int)
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
if args.minimum_training_arm_clusters is not None:
    if not clustered or args.minimum_training_arm_clusters < 1:
        parser.error("minimum training arm clusters requires clustering and a positive count")
    cluster_options["minimum_training_arm_clusters"] = args.minimum_training_arm_clusters
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
oracle_att = []
paired_att_gap = []
known_propensity_att = []
known_outcome_att = []
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
            )["result"]
            contrast = result["contrast"]
        except NativeError as error:
            if (args.weak_overlap or args.cluster_assignment) and "requires both arms" in str(error):
                failed[key] += 1
            elif (args.weak_overlap or args.cluster_assignment) and "falls below minimum_training_arm_" in str(error):
                guarded[key] += 1
            else:
                raise
            continue
        assert contrast["cluster_degrees_of_freedom"] == (cluster_count - 1 if clustered else None)
        estimate = contrast["estimate"]
        if args.cluster_assignment and key == "att":
            def augmented(propensities, predictions):
                return execute(
                    "summarize_causal",
                    method="augmented_treatment_contrast",
                    estimand="att",
                    outcomes=outcomes,
                    treatment=treatment,
                    propensities=propensities,
                    control_predictions=predictions,
                    clusters=cluster_options["clusters"],
                )["result"]["estimate"]

            oracle = augmented(probabilities, baseline)
            known_propensity_att.append(augmented(probabilities, result["control_predictions"]))
            known_outcome_att.append(augmented(result["propensities"], baseline))
            reference = sum(
                (y - y0) if flag else (y0 - y) * p / (1 - p)
                for y, y0, flag, p in zip(outcomes, baseline, treatment, probabilities, strict=True)
            ) / sum(treatment)
            assert abs(oracle - reference) < 1e-12
            oracle_att.append(oracle)
            paired_att_gap.append(estimate - oracle)
        estimates[key].append(estimate)
        distance = abs(estimate - truth)
        error = contrast["influence_standard_error"]
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

if args.cluster_assignment:
    print(f"ATT known-nuisance mean {mean(oracle_att):.6f}, "
          f"paired fitted-minus-known gap {mean(paired_att_gap):.6f}")
    assert len(oracle_att) == len(estimates["att"])
    print(f"ATT known propensity only {mean(known_propensity_att):.6f}, "
          f"known outcome only {mean(known_outcome_att):.6f}")
