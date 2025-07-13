from .base import GrowthCurve
from innovate.backend import current_backend as B

class SymmetricGrowth(GrowthCurve):
    """
    Models symmetric S-shaped growth where the rate of adoption is proportional
    to both the number of adopters and the remaining potential adopters. The
    inflection point is at 50% of the market potential. This is often referred
    to as the Logistic growth model.

    Core Behavior: Growth is driven by internal imitation or simple resource
    constraints. It's a good baseline for simple, internally-driven diffusion.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Calculates the instantaneous growth rate.

        Equation: dN/dt = r * N * (1 - N/K)
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts cumulative adopters over time.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        fun = lambda t, y: self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.flatten()

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate."
            }
        }
