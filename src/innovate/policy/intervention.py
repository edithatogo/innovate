from typing import Callable, Dict, Sequence
from innovate.base.base import DiffusionModel
from innovate.diffuse.bass import BassModel # Example of a model it can modify
import numpy as np

class PolicyIntervention:
    """
    A class to apply policy interventions to a diffusion model.
    """
    def __init__(self, model: DiffusionModel):
        self.model = model
        self._original_params = model.params_.copy() if model.params_ else {}

    def apply_time_varying_params(self, 
                                  t_points: Sequence[float], 
                                  p_effect: Callable[[float], float] = None,
                                  q_effect: Callable[[float], float] = None) -> Callable[[Sequence[float]], Sequence[float]]:
        """
        Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns:
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel): # Extend to other models as needed
            raise TypeError("This policy intervention is currently only supported for BassModel.")

        if not self._original_params:
            raise RuntimeError("Model must be fitted or have initial parameters set before applying policy.")

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get("m", 0.0) # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)
            
            modified_params_at_t_points.append({"p": current_p, "q": current_q, "m": current_m})
        
        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                # Find the closest policy parameters for t_val
                # This is a very basic interpolation/selection
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                temp_model = self.model.__class__() # Create a new instance of the same model type
                temp_model.params_ = modified_params_at_t_points[idx]
                predictions.append(temp_model.predict([t_val])[0]) # Predict for single time point
            return np.array(predictions)

        return predict_with_policy
