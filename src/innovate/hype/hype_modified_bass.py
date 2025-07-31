# src/innovate/hype/hype_modified_bass.py

import numpy as np
from typing import Sequence
from innovate.diffuse.bass import BassModel
from .hype_cycle import HypeCycleModel


class HypeModifiedBassModel:
    """
    A modified Bass model where the adoption parameters (p and q) are
    influenced by a time-varying hype function.
    """

    def __init__(self, bass_model: BassModel, hype_model: HypeCycleModel):
        self.bass_model = bass_model
        self.hype_model = hype_model

    def predict(self, t: Sequence[float], y0: float) -> np.ndarray:
        """
        Predicts the cumulative adoption over time, with hype-modified
        parameters.

        This requires solving the Bass differential equation with time-varying
        p and q.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set."
            )

        from scipy.integrate import odeint

        hype_visibility = self.hype_model.predict(t)
        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        def bass_differential(y, t_step, p_t, q_t):
            return (p_t + q_t * y / m) * (m - y)

        y = np.zeros_like(t, dtype=float)
        y[0] = y0

        for i in range(1, len(t)):
            # Hype influences p and q
            p_t = p_base * (1 + hype_visibility[i - 1])
            q_t = q_base * (1 + hype_visibility[i - 1])

            # Solve for the next step
            y_step = odeint(
                bass_differential, y[i - 1], [t[i - 1], t[i]], args=(p_t, q_t)
            )
            y[i] = y_step[1]

        return y
