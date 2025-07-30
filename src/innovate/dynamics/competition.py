# src/innovate/dynamics/competition.py

from abc import ABC, abstractmethod
from typing import Sequence

class CompetitiveInteraction(ABC):
    """
    Abstract base class for competitive interaction models.
    """

    @abstractmethod
    def compute_interaction_rate(self, population1: float, population2: float, **params):
        """
        Calculates the instantaneous interaction rate.
        """
        pass

class LotkaVolterra(CompetitiveInteraction):
    """
    Implements the Lotka-Volterra competition model.
    """

    def compute_interaction_rate(self, population1: float, population2: float, **params):
        """
        Calculates the instantaneous interaction rate for the Lotka-Volterra model.
        """
        alpha = params.get("alpha", 0.1)
        return alpha * population1 * population2

class MarketShareAttraction(CompetitiveInteraction):
    """
    Implements the market share attraction model.
    """

    def compute_interaction_rate(self, population1: float, population2: float, **params):
        """
        Calculates the instantaneous interaction rate for the market share attraction model.
        """
        attraction1 = params.get("attraction1", 0.1)
        attraction2 = params.get("attraction2", 0.1)
        return attraction1 * population1 - attraction2 * population2

class ReplicatorDynamics(CompetitiveInteraction):
    """
    Implements the replicator dynamics model.
    """

    def compute_interaction_rate(self, population1: float, population2: float, **params):
        """
        Calculates the instantaneous interaction rate for the replicator dynamics model.
        """
        fitness1 = params.get("fitness1", 0.1)
        fitness2 = params.get("fitness2", 0.1)
        average_fitness = (fitness1 * population1 + fitness2 * population2) / (population1 + population2)
        return population1 * (fitness1 - average_fitness)
