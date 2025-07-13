from .base import CompetitiveInteraction
from innovate.backend import current_backend as B

class LotkaVolterraCompetition(CompetitiveInteraction):
    """
    Models the competition between two species using the Lotka-Volterra equations.
    """

    def compute_interaction_rates(self, **params):
        """
        Compute the instantaneous rates of change for two competing species using the Lotka-Volterra competition model.
        
        Parameters:
        	N1 (float): Current population of species 1.
        	N2 (float): Current population of species 2.
        
        Returns:
        	tuple: A pair (dN1dt, dN2dt) representing the rates of change of species 1 and species 2 populations, respectively.
        """
        N1 = params.get("N1")
        N2 = params.get("N2")
        r1 = params.get("growth_rate_1", 0.1)
        r2 = params.get("growth_rate_2", 0.1)
        K1 = params.get("carrying_capacity_1", 1000)
        K2 = params.get("carrying_capacity_2", 1000)
        alpha12 = params.get("competition_coeff_12", 1.0)
        alpha21 = params.get("competition_coeff_21", 1.0)

        dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
        dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
        return dN1dt, dN2dt

    def predict_states(self, time_points, **params):
        """
        Predicts the population trajectories of two competing species over specified time points using the Lotka-Volterra competition model.
        
        Parameters:
            time_points (array-like): Sequence of time points at which to evaluate the populations.
        
        Returns:
            ndarray: Array of shape (len(time_points), 2) containing the predicted populations of both species at each time point.
        """
        from scipy.integrate import solve_ivp

        N1_0 = params.get("N1_0", 1)
        N2_0 = params.get("N2_0", 1)

        fun = lambda t, y: self.compute_interaction_rates(N1=y[0], N2=y[1], **params)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            [N1_0, N2_0],
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Return a dictionary describing the schema for all model parameters, including their types, default values, and descriptions for the Lotka-Volterra competition model.
        """
        return {
            "growth_rate_1": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 1."
            },
            "growth_rate_2": {
                "type": "float",
                "default": 0.1,
                "description": "The intrinsic growth rate of species 2."
            },
            "carrying_capacity_1": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 1."
            },
            "carrying_capacity_2": {
                "type": "float",
                "default": 1000,
                "description": "The carrying capacity of species 2."
            },
            "competition_coeff_12": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 2 on species 1."
            },
            "competition_coeff_21": {
                "type": "float",
                "default": 1.0,
                "description": "The competition coefficient of species 1 on species 2."
            },
            "N1_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 1."
            },
            "N2_0": {
                "type": "float",
                "default": 1,
                "description": "The initial population of species 2."
            }
        }
