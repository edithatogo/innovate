import pytest
import numpy as np
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.policy.intervention import PolicyIntervention


def test_policy_intervention_basic():
    # 1. Create a BassModel instance and fit it with dummy data
    model = BassModel()
    t_data = np.arange(1, 21)
    p_true, q_true, m_true = 0.03, 0.38, 1000
    y_data = model.cumulative_adoption(t_data, p_true, q_true, m_true)
    y_data_noisy = y_data + np.random.normal(0, 10, size=len(t_data))
    y_data_noisy = np.maximum(0, y_data_noisy)

    fitter = ScipyFitter()
    fitter.fit(model, t_data, y_data_noisy)

    # Store original predictions for comparison
    original_predictions = model.predict(t_data)

    # 2. Instantiate PolicyIntervention with the fitted model
    policy_handler = PolicyIntervention(model)

    # 3. Define simple p_effect and q_effect functions
    # Example: p increases by 10% after time 10, q decreases by 5% after time 10
    def p_effect(t):
        return 1.1 if t > 10 else 1.0

    def q_effect(t):
        return 1.05 if t > 10 else 1.0

    # 4. Call apply_time_varying_params and get the predict_with_policy callable
    predict_with_policy = policy_handler.apply_time_varying_params(
        t_points=t_data, p_effect=p_effect, q_effect=q_effect
    )

    # 5. Make predictions with the policy
    policy_predictions = predict_with_policy(t_data)

    # 6. Assert that the predictions are different from the original model's predictions
    # (indicating the policy had an effect)
    # We expect differences, especially after t=10
    assert not np.allclose(original_predictions, policy_predictions, atol=1e-2)

    # More specific check: predictions after policy application should be higher/lower as expected
    # For p increase and q decrease, adoption should generally be higher.
    assert np.all(
        policy_predictions[t_data > 10] > original_predictions[t_data > 10] - 1e-6
    )


def test_policy_intervention_type_error():
    # Test that it raises TypeError for non-BassModel (or other unsupported models)
    from innovate.compete.competition import MultiProductDiffusionModel

    model = MultiProductDiffusionModel(p=[0.1], Q=[[0.1]], m=[100])  # Dummy init
    policy_handler = PolicyIntervention(model)
    with pytest.raises(
        TypeError,
        match="This policy intervention is currently only supported for BassModel.",
    ):
        policy_handler.apply_time_varying_params(t_points=np.arange(10))


def test_policy_intervention_runtime_error_no_params():
    # Test that it raises RuntimeError if model has no parameters set
    model = BassModel()
    policy_handler = PolicyIntervention(model)
    with pytest.raises(
        RuntimeError,
        match="Model must be fitted or have initial parameters set before applying policy.",
    ):
        policy_handler.apply_time_varying_params(t_points=np.arange(10))
