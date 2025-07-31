from innovate.base.base import DiffusionModel, Self
from innovate.backend import current_backend as B
from typing import Sequence, Dict, List, Union
import pandas as pd
import numpy as np


class MultiProductDiffusionModel(DiffusionModel):
    """Generic framework for multi-product/policy diffusion with competition and substitution."""

    def __init__(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[
            Sequence[float]
        ],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i+1}" for i in range(self.N)]

        if not (
            len(self.p) == self.N
            and self.Q.shape == (self.N, self.N)
            and len(self.m) == self.N
        ):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: Dict[str, float] = {}

    def _rhs(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0, y_arr / self.m.flatten(), B.zeros_like(y_arr)
        )
        adoption_share = B.where(
            adoption_share > 1.0, 1.0, adoption_share
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def predict(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m."
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    @staticmethod
    def differential_equation(y, t, params):
        """Differential equations for the multi-product model."""
        p, Q, m = params
        y_arr = B.array(y)
        adoption_share = B.where(
            self.m.flatten() != 0, y_arr / self.m.flatten(), B.zeros_like(y_arr)
        )
        adoption_share = B.where(adoption_share > 1.0, 1.0, adoption_share)
        imitation = B.matmul(Q, adoption_share)
        force = p + imitation
        remaining_potential = B.where(m - y_arr < 0, 0, m - y_arr)
        return force * remaining_potential

    def fit(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def score(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m."
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}"
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, Union[float, List[float], List[List[float]]]]:
        # Return the parameters that were either initialized or fitted
        if self._params:
            return self._params
        else:
            return {"p": self.p.tolist(), "Q": self.Q.tolist(), "m": self.m.tolist()}

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError

    @property
    def param_names(self) -> Sequence[str]:
        return ["p", "Q", "m"]

    def initial_guesses(
        self, t: Sequence[float], y: Sequence[float]
    ) -> Dict[str, float]:
        return {}

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        return {}
