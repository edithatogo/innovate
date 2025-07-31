from .base import ContagionSpread


class SIRModel(ContagionSpread):
    """
    Models the spread of a contagion through a population with Susceptible,
    Infectious, and Recovered states.
    """

    def compute_spread_rate(self, **params):
        """
        Calculates the instantaneous spread rate.

        Equations:
        dS/dt = -beta * S * I
        dI/dt = beta * S * I - gamma * I
        dR/dt = gamma * I

        Compute the instantaneous rates of change for susceptible, infectious, and recovered populations using the SIR model.

        Parameters:
                S (float): Current number of susceptible individuals.
                I (float): Current number of infectious individuals.
                transmission_rate (float, optional): Rate at which the disease spreads (default 0.1).
                recovery_rate (float, optional): Rate at which infectious individuals recover (default 0.01).

        Returns:
                tuple: The rates of change (dS/dt, dI/dt, dR/dt) for susceptible, infectious, and recovered populations.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def predict_states(self, time_points, **params):
        """
        Predicts the states of the population over time.

        Simulate the evolution of susceptible, infectious, and recovered populations over specified time points using the SIR model.

        Parameters:
            time_points (array-like): Sequence of time points at which to compute the population states.
            **params: Optional model parameters, including initial conditions (`S0`, `I0`, `R0`) and rates.

        Returns:
            ndarray: Array of shape (len(time_points), 3) containing the simulated S, I, and R values at each time point.
        """
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.

        Return a dictionary describing the parameter schema for the SIR model, including types, default values, and descriptions for each parameter.
        """
        return {
            "transmission_rate": {
                "type": "float",
                "default": 0.1,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": 0.01,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "I0": {
                "type": "float",
                "default": 1,
                "description": "The initial number of infectious individuals.",
            },
            "R0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of recovered individuals.",
            },
        }
