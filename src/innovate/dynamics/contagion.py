import numpy as np

from .base import ContagionSpread


class SIR(ContagionSpread):
    """Implements the Susceptible-Infected-Recovered (SIR) model."""

    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

    def compute_spread_rate(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def predict_states(self, time_points, **params):
        """Predicts the states of the population over time."""
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
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
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


class SIS(ContagionSpread):
    """Implements the Susceptible-Infected-Susceptible (SIS) model."""

    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

    def compute_spread_rate(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I + gamma * I
        dIdt = beta * S * I - gamma * I
        return dSdt, dIdt

    def predict_states(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        I0 = params.get("I0", 1)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], I=y[1], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, I0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate at which infectious individuals return to susceptible state.",
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
        }


class SEIR(ContagionSpread):
    """Implements the Susceptible-Exposed-Infected-Recovered (SEIR) model."""

    def __init__(self, beta: float = 0.2, sigma: float = 0.5, gamma: float = 0.1):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        S, E, I, R = y
        dSdt = -self.beta * S * I
        dEdt = self.beta * S * I - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])

    def compute_spread_rate(self, **params):
        """Calculates the instantaneous spread rate."""
        S = params.get("S")
        E = params.get("E")
        I = params.get("I")
        beta = params.get("transmission_rate", self.beta)
        sigma = params.get("incubation_rate", self.sigma)
        gamma = params.get("recovery_rate", self.gamma)

        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def predict_states(self, time_points, **params):
        """Predicts the states of the population over time."""
        from scipy.integrate import solve_ivp

        S0 = params.get("S0", 999)
        E0 = params.get("E0", 0)
        I0 = params.get("I0", 1)
        R0 = params.get("R0", 0)

        def ode_func(t, y):
            return self.compute_spread_rate(S=y[0], E=y[1], I=y[2], **params)

        sol = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            [S0, E0, I0, R0],
            t_eval=time_points,
            method="LSODA",
        )
        return sol.y.T

    def get_parameters_schema(self):
        """Returns the schema for the model's parameters."""
        return {
            "transmission_rate": {
                "type": "float",
                "default": self.beta,
                "description": "The rate of transmission of the contagion.",
            },
            "incubation_rate": {
                "type": "float",
                "default": self.sigma,
                "description": "The rate at which exposed individuals become infectious.",
            },
            "recovery_rate": {
                "type": "float",
                "default": self.gamma,
                "description": "The rate of recovery from the contagion.",
            },
            "S0": {
                "type": "float",
                "default": 999,
                "description": "The initial number of susceptible individuals.",
            },
            "E0": {
                "type": "float",
                "default": 0,
                "description": "The initial number of exposed individuals.",
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
