"""Contagion dynamics models."""
from .base import ContagionSpread as ContagionSpread
from .seir import SEIRModel
from .sir import SIRModel
from .sis import SISModel

# For backward compatibility with tests, alias models
SIR = SIRModel
SIS = SISModel
SEIR = SEIRModel
