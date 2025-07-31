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

        Compute the instantaneous rate of change of strategy proportions using the replicator dynamics equation.

        Parameters:
                x (array-like): Current proportions of each strategy.
                payoff_matrix (array-like): Payoff matrix representing interactions between strategies.

        Returns:
                array: The rate of change of each strategy's proportion.
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

        Predicts the evolution of strategy proportions over specified time points using replicator dynamics.

        Parameters:
            time_points (array-like): Sequence of time points at which to evaluate the predicted states.
            x0 (list or array, in params): Initial proportions of each strategy. Must be provided in params.

        Returns:
            ndarray: Array of predicted strategy proportions at each time point, with shape (len(time_points), n_strategies).

        Raises:
            ValueError: If initial proportions `x0` are not provided in params.
        """
        from scipy.integrate import solve_ivp

        x0 = params.get("x0", [])
        if not x0:
            raise ValueError("Initial proportions must be provided.")

        def ode_func(t, y):
            return self.compute_interaction_rates(x=y, **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            x0,
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.

        Return a dictionary describing the expected parameters for the replicator dynamics model, including initial strategy proportions and the payoff matrix.
        """
        return {
            "x0": {
                "type": "list",
                "default": [],
                "description": "A list of initial proportions for each strategy.",
            },
            "payoff_matrix": {
                "type": "list",
                "default": [],
                "description": "The payoff matrix for the game.",
            },
        }
