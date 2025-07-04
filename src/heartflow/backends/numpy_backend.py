import numpy as np
from scipy.integrate import odeint
from typing import Sequence

class NumPyBackend:
    def array(self, data):
        return np.asarray(data)
    
    exp = np.exp
    power = np.power
    def sum(self, a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

    def mean(self, a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
    
    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def solve_ode(self, f, y0: Sequence[float], t: Sequence[float]) -> np.ndarray:
        # scipy.integrate.odeint expects y0 as a 1D array and t as a 1D array
        # The function f should take (y, t, *args) as arguments
        # We need to adapt the signature of f if it expects (t, y, *args)
        # For now, assuming f takes (y, t) as per common scipy usage
        sol = odeint(f, y0, t)
        return sol

    def stack(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(arrays)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def zeros(self, shape: Sequence[int]) -> np.ndarray:
        return np.zeros(shape)
