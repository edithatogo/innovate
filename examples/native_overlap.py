"""Check cross-fitted overlap and contrasts against six hand-worked observations.

These synthetic outcomes do not identify an empirical treatment effect. Meeting
an overlap count threshold does not establish exchangeability, adequate precision
or a suitable analysis design. Threshold one is deliberately small in this demo.
"""

import json
import math

from innovate import NativeError, execute


def main() -> None:
    """Validate held-out predictions, an independent score oracle and overlap guards."""
    payload = {
        "method": "cross_fitted_stratum_ate",
        "outcomes": [1, 2, 3, 4, 5, 6],
        "treatment": [0, 0, 1, 0, 1, 1],
        "strata": ["a"] * 6,
        "folds": [1, 1, 1, 2, 2, 2],
        "minimum_training_arm_count": 1,
    }
    result = execute("summarize_causal", **payload)["result"]

    # Fold one trains on outcomes 4,5,6; fold two trains on outcomes 1,2,3.
    # Their respective treatment fractions are 2/3 and 1/3.
    expected_propensities = [2 / 3] * 3 + [1 / 3] * 3
    expected_control_means = [4.0] * 3 + [1.5] * 3
    expected_treated_means = [5.5] * 3 + [3.0] * 3
    for actual, expected in (
        (result["propensities"], expected_propensities),
        (result["control_predictions"], expected_control_means),
        (result["treated_predictions"], expected_treated_means),
    ):
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected, strict=True))
    overlap = result["overlap"]
    assert overlap["min_training_arm_count"] == 1
    assert overlap["min_training_arm_clusters"] is None
    assert math.isclose(overlap["min_propensity"], 1 / 3)
    assert math.isclose(overlap["max_propensity"], 2 / 3)

    # For score mu1-mu0 + T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e), the six
    # hand-calculated scores are below. This oracle never calls a model fitter.
    scores = [10.5, 7.5, -2.25, -2.25, 7.5, 10.5]
    expected_ate = sum(scores) / len(scores)
    expected_se = math.sqrt(sum((score - expected_ate) ** 2 for score in scores) / (len(scores) * (len(scores) - 1)))
    assert math.isclose(result["contrast"]["estimate"], expected_ate, rel_tol=1e-12)
    assert math.isclose(result["contrast"]["influence_standard_error"], expected_se, rel_tol=1e-12)
    assert result["contrast"]["observations"] == len(scores)

    # Repeated observations within one arm still count as one distinct cluster.
    clusters = ["fold1-control", "fold1-control", "fold1-treated", "fold2-control", "fold2-treated", "fold2-treated"]
    clustered = execute("summarize_causal", **payload, clusters=clusters, minimum_training_arm_clusters=1)["result"]
    assert clustered["overlap"]["min_training_arm_clusters"] == 1
    assert math.isclose(clustered["contrast"]["estimate"], expected_ate, rel_tol=1e-12)
    assert clustered["contrast"]["cluster_count"] == len(set(clusters))

    invalid_designs = [
        ({**payload, "minimum_training_arm_count": 2}, "minimum_training_arm_count"),
        ({**payload, "clusters": clusters, "minimum_training_arm_clusters": 2}, "minimum_training_arm_clusters"),
        (
            {**payload, "clusters": ["split", "c1", "t1", "split", "t2", "t2"]},
            "single cross-fitting fold",
        ),
    ]
    rejected_designs = 0
    for invalid, expected_message in invalid_designs:
        try:
            execute("summarize_causal", **invalid)
        except NativeError as error:
            assert error.code == "invalid_request" and expected_message in str(error)
            rejected_designs += 1
        else:
            raise AssertionError("An invalid overlap or fold design was silently accepted.")
    assert rejected_designs == len(invalid_designs)

    print(
        json.dumps(
            {
                "ate": round(result["contrast"]["estimate"], 6),
                "minimum_training_arm_count": overlap["min_training_arm_count"],
                "minimum_training_arm_clusters": clustered["overlap"]["min_training_arm_clusters"],
                "propensity_range": [round(overlap["min_propensity"], 6), round(overlap["max_propensity"], 6)],
                "rejected_designs": rejected_designs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
