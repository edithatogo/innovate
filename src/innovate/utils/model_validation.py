"""Model validation for the Innovate library."""

from typing import Any

import numpy as np

from .validation import validate_float


def validate_bass_parameters(  # noqa: PLR0912, PLR0915
    params: dict[str, float], t_event: float | None = None
) -> dict[str, Any]:
    """
    Validate Bass model parameters to ensure they result in reasonable behavior.

    Parameters
    ----------
    params : Dict[str, float]
        Dictionary of model parameters
    t_event : Optional[float]
        Time of structural break, if any

    Returns
    -------
    Dict[str, Any]
        Validation results with 'is_valid' flag and 'issues' list
    """
    issues = []
    is_valid = True

    # Check required parameters
    required_params = ["p", "q", "m"]
    if t_event is not None:
        required_params.extend(["p_post", "q_post", "m_post"])

    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        issues.append(f"Missing required parameters: {missing_params}")
        is_valid = False

    # Validate parameter values if they exist
    if "p" in params:
        try:
            p = validate_float(params["p"], "p", min_val=0.0, max_val=1.0)
            if p < 0:
                issues.append(f"Parameter 'p' (innovation coefficient) should be non-negative, got {p}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'p' must be a positive number, got {params['p']}")
            is_valid = False

    if "q" in params:
        try:
            q = validate_float(params["q"], "q", min_val=0.0, max_val=10.0)  # q can be larger than p
            if q < 0:
                issues.append(f"Parameter 'q' (imitation coefficient) should be non-negative, got {q}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'q' must be a positive number, got {params['q']}")
            is_valid = False

    if "m" in params:
        try:
            m = validate_float(params["m"], "m", min_val=0.0)
            if m <= 0:
                issues.append(f"Parameter 'm' (market potential) should be positive, got {m}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'm' must be a positive number, got {params['m']}")
            is_valid = False

    # Validate post-event parameters if they exist
    if "p_post" in params:
        try:
            p_post = validate_float(params["p_post"], "p_post", min_val=0.0, max_val=1.0)
            if p_post < 0:
                issues.append(
                    f"Parameter 'p_post' (post-event innovation coefficient) should be non-negative, got {p_post}"
                )
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'p_post' must be a positive number, got {params['p_post']}")
            is_valid = False

    if "q_post" in params:
        try:
            q_post = validate_float(params["q_post"], "q_post", min_val=0.0, max_val=10.0)
            if q_post < 0:
                issues.append(
                    f"Parameter 'q_post' (post-event imitation coefficient) should be non-negative, got {q_post}"
                )
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'q_post' must be a positive number, got {params['q_post']}")
            is_valid = False

    if "m_post" in params:
        try:
            m_post = validate_float(params["m_post"], "m_post", min_val=0.0)
            if m_post <= 0:
                issues.append(f"Parameter 'm_post' (post-event market potential) should be positive, got {m_post}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'm_post' must be a positive number, got {params['m_post']}")
            is_valid = False

    # Check for reasonable parameter ratios
    if "p" in params and "q" in params:
        p_val = params["p"]
        q_val = params["q"]

        # The ratio p/q affects the timing of the peak adoption rate
        ratio = p_val / q_val if q_val > 0 else float("inf")
        if ratio > 1:  # p > q is unusual but mathematically valid
            issues.append(f"Unusual parameter values: p ({p_val}) > q ({q_val}), typically q > p")

    return {
        "is_valid": is_valid,
        "issues": issues,
        "recommended_action": "Adjust parameters to meet validation criteria" if not is_valid else "No action needed",
    }


def validate_logistic_parameters(params: dict[str, float]) -> dict[str, Any]:
    """
    Validate Logistic model parameters to ensure they result in reasonable behavior.

    Parameters
    ----------
    params : Dict[str, float]
        Dictionary of model parameters

    Returns
    -------
    Dict[str, Any]
        Validation results with 'is_valid' flag and 'issues' list
    """
    issues = []
    is_valid = True

    required_params = ["L", "k", "x0"]
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        issues.append(f"Missing required parameters: {missing_params}")
        is_valid = False

    # Validate parameter values if they exist
    if "L" in params:
        try:
            L = validate_float(params["L"], "L", min_val=0.0)
            if L <= 0:
                issues.append(f"Parameter 'L' (maximum value) should be positive, got {L}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'L' must be a positive number, got {params['L']}")
            is_valid = False

    if "k" in params:
        try:
            k = validate_float(params["k"], "k", min_val=0.0)
            if k <= 0:
                issues.append(f"Parameter 'k' (growth rate) should be positive, got {k}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'k' must be a positive number, got {params['k']}")
            is_valid = False

    if "x0" in params:
        try:
            validate_float(params["x0"], "x0")
        except (ValueError, TypeError):
            issues.append(f"Parameter 'x0' (x-value of sigmoid's midpoint) should be numeric, got {params['x0']}")
            is_valid = False

    return {
        "is_valid": is_valid,
        "issues": issues,
        "recommended_action": "Adjust parameters to meet validation criteria" if not is_valid else "No action needed",
    }


def validate_gompertz_parameters(params: dict[str, float]) -> dict[str, Any]:
    """
    Validate Gompertz model parameters to ensure they result in reasonable behavior.

    Parameters
    ----------
    params : Dict[str, float]
        Dictionary of model parameters

    Returns
    -------
    Dict[str, Any]
        Validation results with 'is_valid' flag and 'issues' list
    """
    issues = []
    is_valid = True

    required_params = ["a", "b", "c"]
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        issues.append(f"Missing required parameters: {missing_params}")
        is_valid = False

    # Validate parameter values if they exist
    if "a" in params:
        try:
            a = validate_float(params["a"], "a", min_val=0.0)
            if a <= 0:
                issues.append(f"Parameter 'a' (upper asymptote) should be positive, got {a}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'a' must be a positive number, got {params['a']}")
            is_valid = False

    if "b" in params:
        try:
            b = validate_float(params["b"], "b", min_val=0.0)
            if b <= 0:
                issues.append(f"Parameter 'b' (displacement along y-axis) should be positive, got {b}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'b' must be a positive number, got {params['b']}")
            is_valid = False

    if "c" in params:
        try:
            c = validate_float(params["c"], "c", min_val=0.0, max_val=1.0)
            if c <= 0:
                issues.append(f"Parameter 'c' (growth rate) should be positive, got {c}")
                is_valid = False
        except (ValueError, TypeError):
            issues.append(f"Parameter 'c' must be a positive number between 0 and 1, got {params['c']}")
            is_valid = False

    return {
        "is_valid": is_valid,
        "issues": issues,
        "recommended_action": "Adjust parameters to meet validation criteria" if not is_valid else "No action needed",
    }


def validate_model_predictions(
    model, t_pred: np.ndarray, y_pred: np.ndarray, max_growth_ratio: float = 0.5
) -> dict[str, Any]:
    """
    Validate that model predictions are reasonable.

    Parameters
    ----------
    model : DiffusionModel
        The fitted model
    t_pred : np.ndarray
        Time points
    y_pred : np.ndarray
        Predicted values
    max_growth_ratio : float
        Maximum allowed ratio of growth between consecutive points relative to current level

    Returns
    -------
    Dict[str, Any]
        Validation results with 'is_valid' flag and 'issues' list
    """
    issues = []
    is_valid = True

    # Check for non-finite values
    if not np.all(np.isfinite(y_pred)):
        issues.append("Model predictions contain non-finite values (NaN or Inf)")
        is_valid = False

    # Check for negative values (unusual for cumulative adoption)
    if np.any(y_pred < 0):
        issues.append(f"Model predictions contain negative values: min = {np.min(y_pred)}")
        is_valid = False

    # Check for reasonable growth rates
    if len(y_pred) > 1:
        diffs = np.diff(y_pred)
        abs_diffs = np.abs(diffs)

        # Calculate relative growth rate (avoid division by zero)
        prev_vals = np.abs(y_pred[:-1])
        prev_vals = np.where(prev_vals == 0, 1e-10, prev_vals)  # Avoid division by zero
        rel_growth = abs_diffs / prev_vals

        if np.any(rel_growth > max_growth_ratio):
            max_idx = np.argmax(rel_growth)
            issues.append(
                f"Unusually high growth rate detected: {rel_growth[max_idx]:.3f} "
                f"between t={t_pred[max_idx]} and t={t_pred[max_idx + 1]}"
            )
            is_valid = False

    # Check if predictions are monotonically increasing (for cumulative models)
    if (
        hasattr(model, "monotonic_check") and model.monotonic_check and not np.all(np.diff(y_pred) >= -1e-10)
    ):  # Allow small numerical errors
        issues.append(
            "Model predictions are not monotonically increasing (cumulative models should generally increase)"
        )
        is_valid = False

    return {
        "is_valid": is_valid,
        "issues": issues,
        "recommended_action": "Adjust model parameters or check for numerical issues"
        if not is_valid
        else "No action needed",
    }
