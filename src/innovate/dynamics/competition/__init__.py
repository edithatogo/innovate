"""Competition dynamics models."""
from .base import CompetitiveInteraction
from .lotka_volterra import LotkaVolterraCompetition
from .market_share_attraction import MarketShareAttraction
from .replicator_dynamics import ReplicatorDynamics

# For backward compatibility with tests, alias LotkaVolterraCompetition to LotkaVolterra
LotkaVolterra = LotkaVolterraCompetition