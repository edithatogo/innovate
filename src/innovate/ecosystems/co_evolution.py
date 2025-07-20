from innovate.base.base import DiffusionModel

class CoEvolutionModel(DiffusionModel):
    def __init__(self):
        self._params = {}

    def fit(self, t, y):
        pass

    def predict(self, t, covariates=None):
        pass

    def differential_equation(self, y, t, p):
        pass
