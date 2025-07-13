from .base import ContagionSpread
from innovate.backend import current_backend as B

class SISModel(ContagionSpread):
    """
    Models the spread of a contagion through a population with Susceptible
    and Infectious states, where recovered individuals can become susceptible again.
    """

    def compute_spread_rate(self, **params):
        """
        Compute the instantaneous rates of change for susceptible and infectious populations in the SIS model.
        
        Parameters:
        	S (float): Current number of susceptible individuals.
        	I (float): Current number of infectious individuals.
        	transmission_rate (float, optional): Probability of transmission per contact (default 0.1).
        	recovery_rate (float, optional): Rate at which infectious individuals become susceptible again (default 0.01).
        
        Returns:
        	tuple: A pair (dSdt, dIdt) representing the rates of change for susceptible and infectious populations, respectively.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def predict_states(self, time_points, **params):
        """
        Simulate and return the evolution of susceptible and infectious populations over specified time points using the SIS model.
        
        Parameters:
            time_points (array-like): Sequence of time points at which to compute the population states.
            S0 (float, optional): Initial number of susceptible individuals. Defaults to 999 if not provided in params.
            I0 (float, optional): Initial number of infectious individuals. Defaults to 1 if not provided in params.
        
        Returns:
            ndarray: Array of shape (len(time_points), 2), where each row contains the susceptible and infectious counts at a given time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        fun = lambda t, y: self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Return a dictionary describing the schema for the SIS model parameters, including types, default values, and descriptions.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion."
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion."
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals."
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals."
            }
        }
