"""Competition dynamics models."""

from .base import CompetitiveInteraction as CompetitiveInteraction
from .lotka_volterra import LotkaVolterraCompetition
from .market_share_attraction import MarketShareAttraction as MarketShareAttraction
from .replicator_dynamics import ReplicatorDynamics as ReplicatorDynamics

# For backward compatibility with tests, alias LotkaVolterraCompetition to LotkaVolterra
LotkaVolterra = LotkaVolterraCompetition
