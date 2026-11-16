"""Scientific integrity tests for fail-closed inference and honest causal dispatch (I01/I02)."""

import polars as pl
import pytest

import innovate


def test_failed_bayesian_inference_raises_and_emits_no_synthetic_draws():
    """I01: Failed Bayesian sampling/grid calls must fail closed, never synthesize draws."""
    bad_payloads = [
        ("bayesian_nuts", {"model": "bass", "time": [], "observed": []}),
        ("bayesian_nuts", {"model": "bass", "time": [1.0, 2.0], "observed": [1.0, float("nan")]}),
        ("bayesian_metropolis", {"model": "bass", "time": [1.0], "observed": [1.0], "sigma": 0.0}),
        ("bayesian_parameter_grid", {"model": "bass", "time": [], "observed": []}),
    ]
    for op, payload in bad_payloads:
        with pytest.raises((innovate.NativeError, ValueError)):
            innovate.execute(op, **payload)


def test_deterministic_fit_has_no_posterior_or_credible_interval_labels():
    """I01: Deterministic point fits must not be labeled as posterior draws."""
    time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    observed = [0.0, 5.0, 15.0, 30.0, 50.0, 70.0]
    fit = innovate.fit("bass", pl.DataFrame({"time": time, "observed": observed}))
    result = fit.envelope["result"]
    for forbidden in (
        "posterior_mean",
        "posterior_means",
        "posterior_draws",
        "credible_interval",
        "credible_intervals",
    ):
        assert forbidden not in result


def test_causal_summary_rejects_unknown_methods_and_unused_fields():
    """I02: Unsupported causal methods, unused fields, and invalid groups must fail closed."""
    with pytest.raises(innovate.NativeError):
        innovate.execute("summarize_causal", method="unsupported_causal_forest")

    # Descriptive contrast returns exact difference without claiming causal adjustment.
    descriptive = innovate.execute(
        "summarize_causal",
        method="treatment_contrast",
        outcomes=[10.0, 12.0, 4.0, 6.0],
        treatment=[1, 1, 0, 0],
    )["result"]
    assert descriptive["overall"]["estimate"] == pytest.approx(6.0)

    # Empty treatment/control groups must fail closed.
    with pytest.raises(innovate.NativeError):
        innovate.execute(
            "summarize_causal",
            method="treatment_contrast",
            outcomes=[1.0, 2.0],
            treatment=[0, 0],
        )

    # Unsupported/unused confounder or option fields must fail closed on matched contrast.
    with pytest.raises(innovate.NativeError):
        innovate.execute(
            "summarize_causal",
            method="matched_treatment_contrast",
            outcomes=[10.0, 12.0, 4.0, 6.0],
            treatment=[1, 1, 0, 0],
            unused_confounder=[1.0, 2.0, 3.0, 4.0],
        )
