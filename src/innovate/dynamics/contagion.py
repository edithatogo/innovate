from .base import ContagionSpread
import numpy as np

class SIR(ContagionSpread):
    """
    Implements the Susceptible-Infected-Recovered (SIR) model.
    """
    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I, R = y
        dSdt = -self.beta * S * I
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I
        return np.array([dSdt, dIdt, dRdt])

class SIS(ContagionSpread):
    """
    Implements the Susceptible-Infected-Susceptible (SIS) model.
    """
    def __init__(self, beta: float = 0.2, gamma: float = 0.1):
        self.beta = beta
        self.gamma = gamma

    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        S, I = y
        dSdt = -self.beta * S * I + self.gamma * I
        dIdt = self.beta * S * I - self.gamma * I
        return np.array([dSdt, dIdt])

class SEIR(ContagionSpread):
    """
    Implements the Susceptible-Exposed-Infected-Recovered (SEIR) model.
    """
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
