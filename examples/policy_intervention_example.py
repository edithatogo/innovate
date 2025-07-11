import numpy as np
import matplotlib.pyplot as plt
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.policy.intervention import PolicyIntervention

def run_policy_intervention_example():
    print("--- Running Policy Intervention Example ---")

    # 1. Generate synthetic Bass model data
    t_data = np.arange(1, 51) # 50 time points
    p_true, q_true, m_true = 0.03, 0.38, 1000
    
    bass_model_true = BassModel()
    # Manually set params for true model to generate data
    bass_model_true.params_ = {"p": p_true, "q": q_true, "m": m_true}
    y_data = bass_model_true.predict(t_data)
    
    # Add some noise for a more realistic fitting scenario
    np.random.seed(42)
    y_data_noisy = y_data + np.random.normal(0, 15, size=len(t_data))
    y_data_noisy = np.maximum(0, y_data_noisy) # Ensure non-negative adoptions

    # 2. Fit a BassModel using ScipyFitter
    fitted_model = BassModel()
    fitter = ScipyFitter()
    fitter.fit(fitted_model, t_data, y_data_noisy)

    print(f"Fitted Bass Model Parameters: {fitted_model.params_}")

    # Get original predictions from the fitted model
    original_predictions = fitted_model.predict(t_data)

    # 3. Define a PolicyIntervention instance
    policy_handler = PolicyIntervention(fitted_model)

    # 4. Define p_effect and q_effect functions
    # Example policy: A temporary boost in p and q between time 20 and 30
    def p_boost_effect(t):
        if 20 <= t <= 30:
            return 1.5 # 50% increase in p
        return 1.0

    def q_boost_effect(t):
        if 20 <= t <= 30:
            return 1.2 # 20% increase in q
        return 1.0

    # 5. Apply the time-varying parameters and get the prediction callable
    predict_with_policy = policy_handler.apply_time_varying_params(
        t_points=t_data,
        p_effect=p_boost_effect,
        q_effect=q_boost_effect
    )

    # 6. Predict with the policy
    policy_predictions = predict_with_policy(t_data)

    # 7. Plot the results
    plt.figure(figsize=(12, 7))
    plt.plot(t_data, y_data_noisy, 'o', label='Observed Data', alpha=0.6)
    plt.plot(t_data, original_predictions, 'b-', label='Fitted Model (No Policy)', linewidth=2)
    plt.plot(t_data, policy_predictions, 'r--', label='Fitted Model (With Policy)', linewidth=2)

    plt.axvspan(20, 30, color='green', alpha=0.1, label='Policy Intervention Period')

    plt.title('Bass Model Diffusion: Policy Intervention Example')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Adoptions')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_policy_intervention_example()
