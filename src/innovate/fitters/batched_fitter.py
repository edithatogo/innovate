from typing import Sequence
from innovate.base.base import DiffusionModel
from innovate.backend import current_backend as B


class BatchedFitter:
    """A fitter class for fitting a model to multiple datasets in a batch."""

    def __init__(self, model: DiffusionModel, fitter):
        self.model = model
        self.fitter = fitter
        self.fitted_params = None

    def fit(
        self, t_batched: Sequence[Sequence[float]], y_batched: Sequence[Sequence[float]]
    ):
        """
        Fits the model to a batch of datasets.

        Args:
            t_batched: A sequence of time sequences.
            y_batched: A sequence of adoption sequences.
        """
        if len(t_batched) != len(y_batched):
            raise ValueError(
                "The number of time sequences and adoption sequences must be the same."
            )

        params_list = []
        for t, y in zip(t_batched, y_batched):
            model_instance = type(self.model)()
            p0 = list(model_instance.initial_guesses(t, y).values())
            bounds = list(zip(*model_instance.bounds(t, y).values()))
            self.fitter.fit(model_instance, t, y, p0=p0, bounds=bounds)
            params_list.append(list(model_instance.params_.values()))

        self.fitted_params = B.array(params_list)
        return self.fitted_params

    def predict(self, t_batched: Sequence[Sequence[float]]):
        """
        Makes predictions for a batch of datasets.

        Args:
            t_batched: A sequence of time sequences.
        """
        if self.fitted_params is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        def predict_single(params, t):
            model_instance = type(self.model)()
            param_dict = dict(zip(model_instance.param_names, params))
            model_instance.params_ = param_dict
            return model_instance.predict(t)

        vmap_predict = B.vmap(predict_single)
        predictions = vmap_predict(self.fitted_params, B.array(t_batched))
        return predictions.reshape(predictions.shape[0], -1)
