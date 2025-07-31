# src/innovate/hype/delayed_hype_bass.py

import numpy as np
from typing import Sequence
from innovate.diffuse.bass import BassModel
from .hype_cycle import HypeCycleModel
from jitcdde import jitcdde, y, t as time
from symengine import exp


class DelayedHypeBassModel:
    """
    A modified Bass model with a time-delayed hype influence, implemented
    using Delay Differential Equations (DDEs).
    """

    def __init__(self, bass_model: BassModel, hype_model: HypeCycleModel, delay: float):
        self.bass_model = bass_model
        self.hype_model = hype_model
        self.delay = delay

    def predict(self, t_eval: Sequence[float], y0: float) -> np.ndarray:
        """
        Predicts the cumulative adoption over time using a DDE solver.
        """
        if not self.bass_model.params_ or not self.hype_model.params_:
            raise RuntimeError(
                "Both the Bass and Hype models must have parameters set."
            )

        p_base = self.bass_model.params_["p"]
        q_base = self.bass_model.params_["q"]
        m = self.bass_model.params_["m"]

        # Define the hype function for the DDE solver
        hype_params = self.hype_model.params_
        k, t0, a_h, t_h, w_h, a_d, t_d, w_d = (
            hype_params["k"],
            hype_params["t0"],
            hype_params["a_hype"],
            hype_params["t_hype"],
            hype_params["w_hype"],
            hype_params["a_d"],
            hype_params["t_d"],
            hype_params["w_d"],
        )

        # Using symengine symbols for the DDE definition
        maturity = 0.5 / (1 + exp(-k * (time - t0)))
        hype = a_h * exp(-((time - t_h) ** 2) / (2 * w_h**2))
        disillusionment = a_d * exp(-((time - t_d) ** 2) / (2 * w_d**2))
        visibility = maturity + hype - disillusionment

        # Define the DDE system
        f = [
            (p_base * (1 + visibility.subs(time, time - self.delay))) * (m - y(0))
            + (q_base * (1 + visibility.subs(time, time - self.delay)))
            * y(0)
            / m
            * (m - y(0))
        ]

        DDE = jitcdde(f)
        DDE.constant_past([y0])
        DDE.step_on_discontinuities()

        adoption = []
        for t_point in t_eval:
            adoption.append(DDE.integrate(t_point)[0])

        return np.array(adoption)
