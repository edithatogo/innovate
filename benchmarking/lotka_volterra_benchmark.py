import time

import numpy as np

from innovate.compete.lotka_volterra import LotkaVolterraModel


def benchmark():
    np.random.seed(42)
    model = LotkaVolterraModel()

    # Generate large dataset to exaggerate the time taken
    t = np.linspace(0, 100, 10000)

    # Dummy params
    model.params_ = {"alpha1": 0.1, "beta1": 0.05, "alpha2": 0.08, "beta2": 0.04}
    y0 = [0.01, 0.01]

    # Warmup
    _ = model.predict_adoption_rate(t[:100], y0)

    start_time = time.time()
    for _ in range(10):
        _ = model.predict_adoption_rate(t, y0)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    benchmark()
