import numpy as np
from innovate.diffuse import BassModel

def main():
    """
    A simple benchmark to reproduce the ValueError in the np.interp function.
    """
    # Create a simple Bass model with a covariate
    covariates = {"price": np.linspace(10, 5, 100)}
    model = BassModel(covariates=list(covariates.keys()))

    # Create some synthetic data
    t = np.linspace(0, 50, 100)

    # Call the differential_equation method with a float t and a numpy array t_eval
    model.differential_equation(t=0.0, y=[1.0], params=[0.001, 0.1, 1000, 0.1, 0.1, 0.1], covariates=covariates, t_eval=t)

if __name__ == "__main__":
    main()
