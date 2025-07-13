from .base import GrowthCurve
from innovate.backend import current_backend as B

class DualInfluenceGrowth(GrowthCurve):
    """
    Models growth driven by two distinct forces: external influence (innovators)
    and internal influence (imitators). The shape of the S-curve can be symmetric
    or asymmetric depending on the relative strength of these two forces. This is
    often referred to as the Bass model.

    Core Behavior: Captures the dynamics of products or ideas that are initially
    pushed by external sources (e.g., advertising) and then take off through
    word-of-mouth or social contagion.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Calculates the instantaneous growth rate.

        Equation: dN/dt = (p + q * (N/M)) * (M - N)
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts cumulative adopters over time.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        fun = lambda t, y: self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

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
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence)."
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence)."
            }
        }
