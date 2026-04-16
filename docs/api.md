# API Reference

This section provides detailed information about the Innovate library's API.

## Stable Entry Points

Prefer the following canonical imports in user-facing code:

```python
from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter
from innovate.compete import MultiProductDiffusionModel, LotkaVolterraModel
from innovate.substitute import CompositeDiffusionModel, FisherPryModel, NortonBassModel
from innovate.ecosystem import ComplementaryGoodsModel
from innovate.backends import use_backend
```

Deep-module imports remain available where needed for compatibility and internal organization.

## Core Modules

### Diffusion Models (`innovate.diffuse`)

#### Bass Model
```{eval-rst}
.. automodule:: innovate.diffuse.bass
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Logistic Model
```{eval-rst}
.. automodule:: innovate.diffuse.logistic
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Gompertz Model
```{eval-rst}
.. automodule:: innovate.diffuse.gompertz
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Classes (`innovate.base`)

#### Diffusion Model Base Class
```{eval-rst}
.. automodule:: innovate.base.base
   :members:
   :undoc-members:
   :show-inheritance:
```

### Fitting Algorithms (`innovate.fitters`)

#### Scipy Fitter
```{eval-rst}
.. automodule:: innovate.fitters.scipy_fitter
   :members:
   :undoc-members:
   :show-inheritance:
```

### Utilities (`innovate.utils`)

#### Validation Utilities
```{eval-rst}
.. automodule:: innovate.utils.validation
   :members:
   :undoc-members:
   :show-inheritance:
```

#### Model Validation Utilities
```{eval-rst}
.. automodule:: innovate.utils.model_validation
   :members:
   :undoc-members:
   :show-inheritance:
```

### Backend Management (`innovate.backends`)

#### Backend Management
```{eval-rst}
.. automodule:: innovate.backends
   :members:
   :undoc-members:
   :show-inheritance:
```
