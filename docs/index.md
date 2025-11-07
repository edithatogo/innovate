# Innovate Library Documentation

Welcome to the documentation for the Innovate library! This library provides tools for modeling innovation and policy diffusion with implementations of classic diffusion models and advanced fitting techniques.

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
api
tutorials
mathematical_background
contributing
```

## About

The Innovate library provides:

- **Diffusion Models**: Classic models like Bass, Logistic, and Gompertz
- **Advanced Fitting**: Scipy, Bayesian, and JAX-based fitting methods
- **Model Competition**: Multi-product diffusion with competitive effects
- **Flexible Parameterization**: Support for covariates and structural breaks
- **Multiple Backends**: NumPy and JAX support for computational efficiency

## Key Features

- **Easy to Use**: Intuitive API similar to scikit-learn
- **Mathematically Rigorous**: Based on ordinary differential equations
- **High Performance**: Optimized for speed and memory usage
- **Extensible**: Easy to add new models and features

## Installation

```bash
pip install innovate
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Example Usage

```python
from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter
import numpy as np

# Sample data
t_data = [0, 1, 2, 3, 4, 5]
y_data = [10, 25, 45, 70, 90, 95]

# Create and fit model
model = BassModel()
fitter = ScipyFitter()
fitted_model = model.fit(fitter, t_data, y_data)

# Make predictions
predictions = fitted_model.predict([6, 7, 8])
print(fitted_model.params_)  # View fitted parameters
```

## Support and Community

- [GitHub Repository](https://github.com/edithatogo/innovate)
- [Issues](https://github.com/edithatogo/innovate/issues)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.