from collections.abc import Callable, Sequence

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.diffuse.bass import BassModel  # Example of a model it can modify
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


class PolicyIntervention:
    """A class to apply policy interventions to a diffusion model."""

    def __init__(self, model: DiffusionModel):
        args = [model]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁPolicyInterventionǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁPolicyInterventionǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁPolicyInterventionǁ__init____mutmut_orig(self, model: DiffusionModel):
        self.model = model
        self._original_params = model.params_.copy() if model.params_ else {}

    def xǁPolicyInterventionǁ__init____mutmut_1(self, model: DiffusionModel):
        self.model = None
        self._original_params = model.params_.copy() if model.params_ else {}

    def xǁPolicyInterventionǁ__init____mutmut_2(self, model: DiffusionModel):
        self.model = model
        self._original_params = None
    
    xǁPolicyInterventionǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁPolicyInterventionǁ__init____mutmut_1': xǁPolicyInterventionǁ__init____mutmut_1, 
        'xǁPolicyInterventionǁ__init____mutmut_2': xǁPolicyInterventionǁ__init____mutmut_2
    }
    xǁPolicyInterventionǁ__init____mutmut_orig.__name__ = 'xǁPolicyInterventionǁ__init__'

    def apply_time_varying_params(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        args = [t_points, p_effect, q_effect]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁPolicyInterventionǁapply_time_varying_params__mutmut_orig'), object.__getattribute__(self, 'xǁPolicyInterventionǁapply_time_varying_params__mutmut_mutants'), args, kwargs, self)

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_orig(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_1(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_2(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                None,
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_3(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "XXThis policy intervention is currently only supported for BassModel.XX",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_4(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "this policy intervention is currently only supported for bassmodel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_5(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "THIS POLICY INTERVENTION IS CURRENTLY ONLY SUPPORTED FOR BASSMODEL.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_6(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_7(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                None,
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_8(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "XXModel must be fitted or have initial parameters set before applying policy.XX",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_9(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_10(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "MODEL MUST BE FITTED OR HAVE INITIAL PARAMETERS SET BEFORE APPLYING POLICY.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_11(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_12(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = None

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_13(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = None
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_14(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = None
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_15(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get(None, 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_16(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", None)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_17(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get(0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_18(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", )
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_19(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("XXpXX", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_20(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("P", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_21(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 1.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_22(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = None
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_23(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get(None, 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_24(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", None)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_25(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get(0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_26(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", )
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_27(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("XXqXX", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_28(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("Q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_29(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 1.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_30(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = None  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_31(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                None,
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_32(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                None,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_33(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_34(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_35(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "XXmXX",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_36(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "M",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_37(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                1.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_38(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p = p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_39(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p /= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_40(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(None)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_41(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q = q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_42(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q /= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_43(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(None)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_44(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                None,
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_45(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"XXpXX": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_46(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"P": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_47(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "XXqXX": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_48(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "Q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_49(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "XXmXX": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_50(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "M": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_51(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = None
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_52(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = None
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_53(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(None)
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_54(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(None))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_55(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) + t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_56(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(None) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_57(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = None
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_58(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = None
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_59(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["XXpXX"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_60(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["P"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_61(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["XXqXX"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_62(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["Q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_63(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["XXmXX"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_64(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["M"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_65(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = None
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_66(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(None)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_67(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) / t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_68(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(+(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_69(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p - q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_70(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p != 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_71(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 1:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_72(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = None
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_73(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m / (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_74(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 + np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_75(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (2 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_76(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(None))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_77(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q / t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_78(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(+q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_79(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = None

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_80(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) * (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_81(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m / (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_82(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 + expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_83(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (2 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_84(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 - (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_85(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (2 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_86(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) / expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_87(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q * p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_88(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(None, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_89(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, None):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_90(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_91(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, ):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_92(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 1):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_93(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(None, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_94(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, None):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_95(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_96(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, ):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_97(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 1):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_98(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = None
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_99(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 1.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_100(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = None
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_101(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(None)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_102(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q / t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_103(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(+q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_104(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = None
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_105(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m / (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_106(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 + expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_107(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (2 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_108(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = None
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_109(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(None)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_110(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) / t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_111(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(+(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_112(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p - q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_113(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = None
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_114(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) * (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_115(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m / (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_116(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 + expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_117(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (2 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_118(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 - (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_119(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (2 + (q / p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_120(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) / expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_121(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q * p) * expo)
                predictions.append(pred)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_122(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(None)
            return np.array(predictions)

        return predict_with_policy

    def xǁPolicyInterventionǁapply_time_varying_params__mutmut_123(
        self,
        t_points: Sequence[float],
        p_effect: Callable[[float], float] | None = None,
        q_effect: Callable[[float], float] | None = None,
    ) -> Callable[[Sequence[float]], Sequence[float]]:
        """Applies time-varying effects to 'p' and 'q' parameters of the model.
        This method is specifically designed for Bass-like models.

        Args:
        ----
            t_points: A sequence of time points for which to apply the effects.
            p_effect: A callable that takes time (float) and returns a multiplier for 'p'.
            q_effect: A callable that takes time (float) and returns a multiplier for 'q'.

        Returns
        -------
            A callable that takes a sequence of time points and returns predictions
            with the applied time-varying policy effects.
        """
        if not isinstance(self.model, BassModel):  # Extend to other models as needed
            raise TypeError(
                "This policy intervention is currently only supported for BassModel.",
            )

        if not self._original_params:
            raise RuntimeError(
                "Model must be fitted or have initial parameters set before applying policy.",
            )

        # Store original parameters if not already done
        if not self._original_params:
            self._original_params = self.model.params_.copy()

        # Pre-calculate modified parameters for each t_point
        modified_params_at_t_points = []
        for t in t_points:
            current_p = self._original_params.get("p", 0.0)
            current_q = self._original_params.get("q", 0.0)
            current_m = self._original_params.get(
                "m",
                0.0,
            )  # m is assumed constant for this policy

            if p_effect:
                current_p *= p_effect(t)
            if q_effect:
                current_q *= q_effect(t)

            modified_params_at_t_points.append(
                {"p": current_p, "q": current_q, "m": current_m},
            )

        # Create a callable that predicts with policy effects
        def predict_with_policy(t_eval: Sequence[float]) -> Sequence[float]:
            predictions = []
            for t_val in t_eval:
                idx = np.argmin(np.abs(np.array(t_points) - t_val))
                params = modified_params_at_t_points[idx]
                p, q, m = params["p"], params["q"], params["m"]
                expo = np.exp(-(p + q) * t_val)
                # Avoid division by zero when p is 0
                if p == 0:
                    pred = m * (1 - np.exp(-q * t_val))
                else:
                    pred = m * (1 - expo) / (1 + (q / p) * expo)

                # Handle edge case when p is zero to avoid division by zero
                if np.isclose(p, 0):
                    # When p≈0, use limit form or return 0 if both p and q are ≈0
                    if np.isclose(q, 0):
                        pred = 0.0
                    else:
                        # L'Hôpital's rule: lim(p->0) of Bass model formula
                        expo = np.exp(-q * t_val)
                        pred = m * (1 - expo)
                else:
                    expo = np.exp(-(p + q) * t_val)
                    pred = m * (1 - expo) / (1 + (q / p) * expo)
                predictions.append(pred)
            return np.array(None)

        return predict_with_policy
    
    xǁPolicyInterventionǁapply_time_varying_params__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁPolicyInterventionǁapply_time_varying_params__mutmut_1': xǁPolicyInterventionǁapply_time_varying_params__mutmut_1, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_2': xǁPolicyInterventionǁapply_time_varying_params__mutmut_2, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_3': xǁPolicyInterventionǁapply_time_varying_params__mutmut_3, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_4': xǁPolicyInterventionǁapply_time_varying_params__mutmut_4, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_5': xǁPolicyInterventionǁapply_time_varying_params__mutmut_5, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_6': xǁPolicyInterventionǁapply_time_varying_params__mutmut_6, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_7': xǁPolicyInterventionǁapply_time_varying_params__mutmut_7, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_8': xǁPolicyInterventionǁapply_time_varying_params__mutmut_8, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_9': xǁPolicyInterventionǁapply_time_varying_params__mutmut_9, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_10': xǁPolicyInterventionǁapply_time_varying_params__mutmut_10, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_11': xǁPolicyInterventionǁapply_time_varying_params__mutmut_11, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_12': xǁPolicyInterventionǁapply_time_varying_params__mutmut_12, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_13': xǁPolicyInterventionǁapply_time_varying_params__mutmut_13, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_14': xǁPolicyInterventionǁapply_time_varying_params__mutmut_14, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_15': xǁPolicyInterventionǁapply_time_varying_params__mutmut_15, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_16': xǁPolicyInterventionǁapply_time_varying_params__mutmut_16, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_17': xǁPolicyInterventionǁapply_time_varying_params__mutmut_17, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_18': xǁPolicyInterventionǁapply_time_varying_params__mutmut_18, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_19': xǁPolicyInterventionǁapply_time_varying_params__mutmut_19, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_20': xǁPolicyInterventionǁapply_time_varying_params__mutmut_20, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_21': xǁPolicyInterventionǁapply_time_varying_params__mutmut_21, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_22': xǁPolicyInterventionǁapply_time_varying_params__mutmut_22, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_23': xǁPolicyInterventionǁapply_time_varying_params__mutmut_23, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_24': xǁPolicyInterventionǁapply_time_varying_params__mutmut_24, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_25': xǁPolicyInterventionǁapply_time_varying_params__mutmut_25, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_26': xǁPolicyInterventionǁapply_time_varying_params__mutmut_26, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_27': xǁPolicyInterventionǁapply_time_varying_params__mutmut_27, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_28': xǁPolicyInterventionǁapply_time_varying_params__mutmut_28, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_29': xǁPolicyInterventionǁapply_time_varying_params__mutmut_29, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_30': xǁPolicyInterventionǁapply_time_varying_params__mutmut_30, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_31': xǁPolicyInterventionǁapply_time_varying_params__mutmut_31, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_32': xǁPolicyInterventionǁapply_time_varying_params__mutmut_32, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_33': xǁPolicyInterventionǁapply_time_varying_params__mutmut_33, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_34': xǁPolicyInterventionǁapply_time_varying_params__mutmut_34, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_35': xǁPolicyInterventionǁapply_time_varying_params__mutmut_35, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_36': xǁPolicyInterventionǁapply_time_varying_params__mutmut_36, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_37': xǁPolicyInterventionǁapply_time_varying_params__mutmut_37, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_38': xǁPolicyInterventionǁapply_time_varying_params__mutmut_38, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_39': xǁPolicyInterventionǁapply_time_varying_params__mutmut_39, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_40': xǁPolicyInterventionǁapply_time_varying_params__mutmut_40, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_41': xǁPolicyInterventionǁapply_time_varying_params__mutmut_41, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_42': xǁPolicyInterventionǁapply_time_varying_params__mutmut_42, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_43': xǁPolicyInterventionǁapply_time_varying_params__mutmut_43, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_44': xǁPolicyInterventionǁapply_time_varying_params__mutmut_44, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_45': xǁPolicyInterventionǁapply_time_varying_params__mutmut_45, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_46': xǁPolicyInterventionǁapply_time_varying_params__mutmut_46, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_47': xǁPolicyInterventionǁapply_time_varying_params__mutmut_47, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_48': xǁPolicyInterventionǁapply_time_varying_params__mutmut_48, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_49': xǁPolicyInterventionǁapply_time_varying_params__mutmut_49, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_50': xǁPolicyInterventionǁapply_time_varying_params__mutmut_50, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_51': xǁPolicyInterventionǁapply_time_varying_params__mutmut_51, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_52': xǁPolicyInterventionǁapply_time_varying_params__mutmut_52, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_53': xǁPolicyInterventionǁapply_time_varying_params__mutmut_53, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_54': xǁPolicyInterventionǁapply_time_varying_params__mutmut_54, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_55': xǁPolicyInterventionǁapply_time_varying_params__mutmut_55, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_56': xǁPolicyInterventionǁapply_time_varying_params__mutmut_56, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_57': xǁPolicyInterventionǁapply_time_varying_params__mutmut_57, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_58': xǁPolicyInterventionǁapply_time_varying_params__mutmut_58, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_59': xǁPolicyInterventionǁapply_time_varying_params__mutmut_59, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_60': xǁPolicyInterventionǁapply_time_varying_params__mutmut_60, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_61': xǁPolicyInterventionǁapply_time_varying_params__mutmut_61, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_62': xǁPolicyInterventionǁapply_time_varying_params__mutmut_62, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_63': xǁPolicyInterventionǁapply_time_varying_params__mutmut_63, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_64': xǁPolicyInterventionǁapply_time_varying_params__mutmut_64, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_65': xǁPolicyInterventionǁapply_time_varying_params__mutmut_65, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_66': xǁPolicyInterventionǁapply_time_varying_params__mutmut_66, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_67': xǁPolicyInterventionǁapply_time_varying_params__mutmut_67, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_68': xǁPolicyInterventionǁapply_time_varying_params__mutmut_68, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_69': xǁPolicyInterventionǁapply_time_varying_params__mutmut_69, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_70': xǁPolicyInterventionǁapply_time_varying_params__mutmut_70, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_71': xǁPolicyInterventionǁapply_time_varying_params__mutmut_71, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_72': xǁPolicyInterventionǁapply_time_varying_params__mutmut_72, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_73': xǁPolicyInterventionǁapply_time_varying_params__mutmut_73, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_74': xǁPolicyInterventionǁapply_time_varying_params__mutmut_74, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_75': xǁPolicyInterventionǁapply_time_varying_params__mutmut_75, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_76': xǁPolicyInterventionǁapply_time_varying_params__mutmut_76, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_77': xǁPolicyInterventionǁapply_time_varying_params__mutmut_77, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_78': xǁPolicyInterventionǁapply_time_varying_params__mutmut_78, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_79': xǁPolicyInterventionǁapply_time_varying_params__mutmut_79, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_80': xǁPolicyInterventionǁapply_time_varying_params__mutmut_80, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_81': xǁPolicyInterventionǁapply_time_varying_params__mutmut_81, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_82': xǁPolicyInterventionǁapply_time_varying_params__mutmut_82, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_83': xǁPolicyInterventionǁapply_time_varying_params__mutmut_83, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_84': xǁPolicyInterventionǁapply_time_varying_params__mutmut_84, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_85': xǁPolicyInterventionǁapply_time_varying_params__mutmut_85, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_86': xǁPolicyInterventionǁapply_time_varying_params__mutmut_86, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_87': xǁPolicyInterventionǁapply_time_varying_params__mutmut_87, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_88': xǁPolicyInterventionǁapply_time_varying_params__mutmut_88, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_89': xǁPolicyInterventionǁapply_time_varying_params__mutmut_89, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_90': xǁPolicyInterventionǁapply_time_varying_params__mutmut_90, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_91': xǁPolicyInterventionǁapply_time_varying_params__mutmut_91, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_92': xǁPolicyInterventionǁapply_time_varying_params__mutmut_92, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_93': xǁPolicyInterventionǁapply_time_varying_params__mutmut_93, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_94': xǁPolicyInterventionǁapply_time_varying_params__mutmut_94, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_95': xǁPolicyInterventionǁapply_time_varying_params__mutmut_95, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_96': xǁPolicyInterventionǁapply_time_varying_params__mutmut_96, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_97': xǁPolicyInterventionǁapply_time_varying_params__mutmut_97, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_98': xǁPolicyInterventionǁapply_time_varying_params__mutmut_98, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_99': xǁPolicyInterventionǁapply_time_varying_params__mutmut_99, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_100': xǁPolicyInterventionǁapply_time_varying_params__mutmut_100, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_101': xǁPolicyInterventionǁapply_time_varying_params__mutmut_101, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_102': xǁPolicyInterventionǁapply_time_varying_params__mutmut_102, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_103': xǁPolicyInterventionǁapply_time_varying_params__mutmut_103, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_104': xǁPolicyInterventionǁapply_time_varying_params__mutmut_104, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_105': xǁPolicyInterventionǁapply_time_varying_params__mutmut_105, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_106': xǁPolicyInterventionǁapply_time_varying_params__mutmut_106, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_107': xǁPolicyInterventionǁapply_time_varying_params__mutmut_107, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_108': xǁPolicyInterventionǁapply_time_varying_params__mutmut_108, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_109': xǁPolicyInterventionǁapply_time_varying_params__mutmut_109, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_110': xǁPolicyInterventionǁapply_time_varying_params__mutmut_110, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_111': xǁPolicyInterventionǁapply_time_varying_params__mutmut_111, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_112': xǁPolicyInterventionǁapply_time_varying_params__mutmut_112, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_113': xǁPolicyInterventionǁapply_time_varying_params__mutmut_113, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_114': xǁPolicyInterventionǁapply_time_varying_params__mutmut_114, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_115': xǁPolicyInterventionǁapply_time_varying_params__mutmut_115, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_116': xǁPolicyInterventionǁapply_time_varying_params__mutmut_116, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_117': xǁPolicyInterventionǁapply_time_varying_params__mutmut_117, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_118': xǁPolicyInterventionǁapply_time_varying_params__mutmut_118, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_119': xǁPolicyInterventionǁapply_time_varying_params__mutmut_119, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_120': xǁPolicyInterventionǁapply_time_varying_params__mutmut_120, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_121': xǁPolicyInterventionǁapply_time_varying_params__mutmut_121, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_122': xǁPolicyInterventionǁapply_time_varying_params__mutmut_122, 
        'xǁPolicyInterventionǁapply_time_varying_params__mutmut_123': xǁPolicyInterventionǁapply_time_varying_params__mutmut_123
    }
    xǁPolicyInterventionǁapply_time_varying_params__mutmut_orig.__name__ = 'xǁPolicyInterventionǁapply_time_varying_params'
