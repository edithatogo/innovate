from typing import Sequence, Dict

class JaxFitter:
    """A fitter class that will use JAX for model estimation (Phase 2)."""

    def fit(self, model, t: Sequence[float], y: Sequence[float], **kwargs) -> Dict[str, float]:
        raise NotImplementedError("JAX Fitter not implemented in Phase 1. Will be available in Phase 2.")
