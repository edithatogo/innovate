import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict
from innovate.base.base import DiffusionModel

def estimate_bass_mom(t: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """
    Estimates the parameters (p, q, m) of the Bass Diffusion Model using the Method of Moments.
    This implementation uses a linear regression approach based on incremental adoptions.

    Args:
        t: A sequence of time points.
        y: A sequence of cumulative adoptions corresponding to the time points.

    Returns:
        A tuple (p, q, m) representing the estimated parameters.
    """
    if len(t) != len(y) or len(t) < 3:
        raise ValueError("Input sequences t and y must have the same length and at least 3 data points.")

    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, index=t)

    # Calculate incremental adoptions (x_t)
    # The first incremental adoption is y[0] if y starts from 0, or y[1]-y[0] if y[0] is the first cumulative.
    # For simplicity, let's use diff() on the series.
    incremental_adoptions = y_series.diff().fillna(y_series.iloc[0])

    # Create lagged cumulative adoptions (y_{t-1})
    lagged_cumulative_adoptions = y_series.shift(1).fillna(0)

    # Prepare data for linear regression: x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # We need to exclude the first point where lagged_cumulative_adoptions is 0, as it's not a true lagged value.
    data_for_reg = pd.DataFrame({
        'x_t': incremental_adoptions,
        'y_t_minus_1': lagged_cumulative_adoptions
    })

    # Remove the first row as y_t_minus_1 is 0 (or NaN if using dropna)
    data_for_reg = data_for_reg.iloc[1:].dropna()

    if len(data_for_reg) < 3:
        raise ValueError("Not enough valid data points for Bass MoM estimation after preprocessing.")

    X = pd.DataFrame({
        'intercept': 1,
        'y_t_minus_1': data_for_reg['y_t_minus_1'],
        'y_t_minus_1_sq': data_for_reg['y_t_minus_1']**2
    })
    y_reg = data_for_reg['x_t']

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Linear regression for Bass MoM failed: {e}. Check data for collinearity or insufficient variation.")

    a, b, c = beta[0], beta[1], beta[2]

    # Solve for p, q, m
    # From x_t = a + b * y_{t-1} + c * y_{t-1}^2
    # And Bass model incremental form: x_t = p*m + (q-p)*y_{t-1} - (q/m)*y_{t-1}^2
    # Comparing coefficients:
    # c = -q/m  => q = -c*m
    # b = q - p
    # a = p*m

    # Rearrange to a quadratic equation for m: c*m^2 + b*m + a = 0
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("Discriminant is negative, no real solution for m. Bass MoM estimation failed. Data might not fit Bass model assumptions well.")

    m1 = (-b + np.sqrt(discriminant)) / (2 * c)
    m2 = (-b - np.sqrt(discriminant)) / (2 * c)

    # Choose the positive and meaningful m (market potential should be positive)
    # And m should be greater than or equal to the maximum observed cumulative adoption.
    m_candidates = [val for val in [m1, m2] if val > 0 and val >= np.max(y)]

    if not m_candidates:
        raise ValueError("No valid positive market potential (m) found that is greater than or equal to max observed adoption. Bass MoM estimation failed.")
    
    # If there are two valid candidates, typically the larger one is chosen for m.
    m = max(m_candidates)

    # Calculate q and p
    q = -c * m
    p = a / m

    # Ensure p and q are positive, as per Bass model assumptions
    if p <= 0 or q <= 0:
        raise ValueError(f"Estimated p ({p}) or q ({q}) is not positive. Bass MoM estimation failed. Data might not fit Bass model assumptions well.")

    return p, q, m

class MoMFitter:
    """
    Fitter for the Bass Diffusion Model using the Method of Moments (MoM).
    This fitter is specifically designed for the BassModel.
    """
    def __init__(self):
        self._params: Dict[str, float] = {}

    def fit(self, model: DiffusionModel, t: Sequence[float], y: Sequence[float]) -> DiffusionModel:
        """
        Fits the BassModel using the Method of Moments.

        Args:
            model: An instance of BassModel.
            t: Time points.
            y: Cumulative adoption data.

        Returns:
            The fitted BassModel instance.
        """
        # Ensure the model is a BassModel instance
        from innovate.diffuse.bass import BassModel # Import here to avoid circular dependency
        if not isinstance(model, BassModel):
            raise TypeError("MoMFitter can only fit BassModel instances.")

        p, q, m = estimate_bass_mom(t, y)
        model.params_ = {"p": p, "q": q, "m": m}
        self._params = model.params_ # Store fitted parameters internally
        return model

    @property
    def params_(self) -> Dict[str, float]:
        return self._params