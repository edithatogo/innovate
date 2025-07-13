from .base import CompetitiveInteraction
from innovate.backend import current_backend as B

class ReplicatorDynamics(CompetitiveInteraction):
    """
    Models the evolution of strategy proportions based on relative fitness/payoff in a game.
    """

    def compute_interaction_rates(self, **params):
        """
        Calculates the instantaneous interaction rates.

        Equation: dxi/dt = xi * (Ui(x) - U_bar(x))
        """
        x = params.get("x")
        payoff_matrix = params.get("payoff_matrix")

        U = B.matmul(B.array(payoff_matrix), B.array(x))
        U_bar = B.sum(B.array(x) * U)

        dxdt = B.array(x) * (U - U_bar)
        return dxdt

    def predict_states(self, time_points, **params):
        """
        Predicts the states of the competing entities over time.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        fun = lambda t, y: self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy."
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game."
            }
        }
