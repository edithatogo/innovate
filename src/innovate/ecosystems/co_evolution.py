from innovate.base.base import DiffusionModel


class CoEvolutionModel(DiffusionModel):
    def __init__(self):
        self._params = {}

    def fit(self, t, y):
        pass

    def predict(self, t, covariates=None):
        pass

    @staticmethod
    def differential_equation(t, y, p):
        pass
