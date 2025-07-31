from .base import GrowthCurve


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

        Calculate the instantaneous growth rate for symmetric (logistic) growth.

        Parameters:
                current_adopters (float): The current number of adopters.
                total_potential (float): The total potential number of adopters.

        Returns:
                float: The rate of change in adopters at the current state, or 0 if total potential is zero or negative.
        """
        r = params.get("growth_rate", 0.1)
        K = total_potential
        N = current_adopters
        return r * N * (1 - N / K) if K > 0 else 0

    def predict_cumulative(
        self, time_points, initial_adopters, total_potential, **params
    ):
        """
        Predicts cumulative adopters over time.

        Predicts the cumulative number of adopters at specified time points using the logistic growth model.

        Parameters:
            time_points (array-like): Sequence of time points at which to predict cumulative adopters.
            initial_adopters (float): Initial number of adopters at the start of the prediction period.
            total_potential (float): Total potential number of adopters (carrying capacity).

        Returns:
            numpy.ndarray: Array of predicted cumulative adopters corresponding to each time point.
        """
        from scipy.integrate import solve_ivp

        r = params.get("growth_rate", 0.1)
        K = total_potential

        def ode_func(t, y):
            return self.compute_growth_rate(y, K, growth_rate=r)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.flatten()

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the model's parameters, including type, default value, and description for each parameter.
        """
        return {
            "growth_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate.",
            }
        }
