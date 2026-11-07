"""Check the native augmented ATE with known and misspecified nuisance inputs."""

from random import Random
from statistics import mean

from innovate import execute

rng = Random(982)
control = [0.0, 4.0] * 200
treated = [2.0, 6.0] * 200
propensities = [0.2, 0.8] * 200
estimates = [[], [], [], []]
for _ in range(400):
    treatment = [int(rng.random() < p) for p in propensities]
    outcomes = [
        (y1 if flag else y0) + rng.gauss(0, 1)
        for y0, y1, flag in zip(control, treated, treatment, strict=True)
    ]
    for draws, good_p, good_y in zip(
        estimates, (True, True, False, False), (True, False, True, False), strict=True
    ):
        result = execute(
            "summarize_causal",
            method="augmented_treatment_contrast",
            outcomes=outcomes,
            treatment=treatment,
            propensities=propensities if good_p else [0.5] * len(outcomes),
            treated_predictions=treated if good_y else [2.0] * len(outcomes),
            control_predictions=control if good_y else [0.0] * len(outcomes),
        )["result"]
        draws.append(result["estimate"])
means = [mean(draws) for draws in estimates]
print("Both correct, propensity only, outcome only, neither:", means)
assert all(abs(estimate - 2.0) < 0.05 for estimate in means[:3])
# With both nuisance inputs wrong, confounding adds 4 * (0.8 - 0.2).
assert abs(means[3] - 4.4) < 0.05
