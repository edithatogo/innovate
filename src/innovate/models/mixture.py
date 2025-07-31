# src/innovate/models/mixture.py

from innovate.base.base import DiffusionModel
from typing import Sequence, Dict, List, Type
from innovate.backend import current_backend as B
from innovate.fitters.scipy_fitter import ScipyFitter
import numpy as np

class MixtureModel(DiffusionModel):
    """
    A latent-class mixture model for diffusion dynamics.

    This model identifies distinct adopter segments from the data by fitting
    multiple diffusion submodels simultaneously. It uses the Expectation-
    Maximization (EM) algorithm to infer both the parameters of each submodel
    and the probability that each data point belongs to a particular segment.

    Parameters
    ----------
    model_classes : Sequence[Type[DiffusionModel]]
        A list of diffusion model classes (e.g., [Bass, Gompertz]) to use as
        the components of the mixture.
    max_iter : int, optional
        The maximum number of iterations for the EM algorithm (default is 100).
    tol : float, optional
        The tolerance for convergence of the log-likelihood (default is 1e-6).
    """

    def __init__(self, model_classes: Sequence[Type[DiffusionModel]], max_iter: int = 100, tol: float = 1e-6):
        self.model_classes = model_classes
        self.num_components = len(model_classes)
        self.models = [cls() for cls in model_classes]
        self.weights = B.ones(self.num_components) / self.num_components
        self.max_iter = max_iter
        self.tol = tol
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        """The names of the model parameters."""
        names: List[str] = []
        for i, model in enumerate(self.models):
            for pname in model.param_names:
                names.append(f"model_{i}_{pname}")
        for i in range(self.num_components):
            names.append(f"weight_{i}")
        return names

    def fit(self, t: Sequence[float], y: Sequence[float]):
        """
        Fits the mixture model to the data using Expectation-Maximization.

        Parameters
        ----------
        t : Sequence[float]
            A sequence of time points.
        y : Sequence[float]
            A sequence of observed data.
        """
        t_arr = B.array(t)
        y_arr = B.array(y)
        n_samples = len(y_arr)
        
        # --- Initialization ---
        # Initialize model parameters by fitting each model to the whole dataset
        fitter = ScipyFitter()
        for model in self.models:
            fitter.fit(model, t_arr, y_arr)

        log_likelihood = -np.inf

        for it in range(self.max_iter):
            # --- E-step: Calculate responsibilities ---
            component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
            # Add a small epsilon to avoid log(0)
            weighted_preds = B.log(component_preds + 1e-9) + B.log(self.weights[:, None])
            
            # Responsibilities (gamma_nk)
            log_responsibilities = weighted_preds - B.logsumexp(weighted_preds, axis=0)
            responsibilities = B.exp(log_responsibilities)

            # --- M-step: Update parameters and weights ---
            # Update weights
            self.weights = B.mean(responsibilities, axis=1)

            # Update model parameters with a weighted fit
            for k in range(self.num_components):
                w = responsibilities[k, :] + 1e-9 # Add epsilon to avoid zero weights
                try:
                    fitter.fit(self.models[k], t_arr, y_arr, weights=w)
                except RuntimeError:
                    # If fitting fails, keep old parameters
                    pass

            # --- Check for convergence ---
            new_log_likelihood = B.sum(B.logsumexp(weighted_preds, axis=0))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self._update_params_from_models()
        return self

    def _update_params_from_models(self):
        """Internal helper to populate the main params_ dictionary."""
        self._params = {}
        for i, model in enumerate(self.models):
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        for i, w in enumerate(self.weights):
            self._params[f"weight_{i}"] = w

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        """
        Makes predictions using the fitted mixture model.
        """
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        t_arr = B.array(t)
        component_preds = B.stack([B.array(m.predict(t_arr)) for m in self.models])
        
        # Weighted average of the component predictions
        y_pred = B.sum(component_preds * self.weights[:, None], axis=0)
        return y_pred

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        """Sets the model parameters and updates the internal models."""
        self._params = value
        # Update weights
        self.weights = B.array([value.get(f"weight_{i}", 0) for i in range(self.num_components)])
        # Update submodel parameters
        for i, model in enumerate(self.models):
            prefix = f"model_{i}_"
            model_params = {k[len(prefix)]:] v for k, v in value.items() if k.startswith(prefix)}
            model.params_ = model_params

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        """
        Calculates the R-squared score for the model.
        """
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def __repr__(self):
        return f"MixtureModel(models={self.model_classes}, weights={self.weights})"