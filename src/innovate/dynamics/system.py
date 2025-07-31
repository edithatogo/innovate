# src/innovate/dynamics/system.py

from abc import ABC, abstractmethod


class SystemBehavior(ABC):
    """
    Abstract base class for system behavior models.
    """

    @abstractmethod
    def compute_system_rate(self, **params):
        """
        Calculates the instantaneous system rate.
        """
        pass
