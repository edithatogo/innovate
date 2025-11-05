"""Contagion dynamics models."""
from .base import ContagionSpread
from .sir import SIRModel
from .sis import SISModel
from .seir import SEIRModel

# For backward compatibility with tests, alias models
SIR = SIRModel
SIS = SISModel
SEIR = SEIRModel