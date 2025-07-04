from .base import DiffusionModel, Self
from ..backend import current_backend as B
from typing import Sequence, Dict, List, Union
import pandas as pd
import numpy as np

class MultiProductDiffusionModel(DiffusionModel):
    """Generic framework for multi-product/policy diffusion with competition and substitution."""

    def __init__(self,
                 p: Sequence[float],       # length N: intrinsic adoption rates
                 Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
                 m: Sequence[float],       # length N: ultimate market potentials
                 names: Sequence[str] = None):
        
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i+1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._fitted_params: Dict[str, float] = {}

    def _rhs(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])
        
        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(self.m != 0, y_arr / self.m, B.zeros(self.N))
        adoption_share = B.where(adoption_share > 1.0, 1.0, adoption_share) # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation                  # shape (N,)
        
        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def fit(self, t: Sequence[float], data: pd.DataFrame) -> Self:
        from scipy.optimize import curve_fit

        t_arr = np.array(t)
        # Ensure data contains all product names and is in the correct order
        if not all(name in data.columns for name in self.names):
            raise ValueError(f"Dataframe must contain columns for all products: {self.names}")
        
        # Extract observed data for the products in the specified order
        y_obs_df = data[list(self.names)]
        y_obs_flat = y_obs_df.values.flatten() # Flatten for curve_fit

        # Define a wrapper function for curve_fit that matches its signature
        def _predict_wrapper(t_points, *params_flat):
            # Reshape params_flat back into p, Q, m
            # This is a simplified approach for initial fitting. 
            # A more robust approach would involve a custom fitter or optimization routine.
            # For now, we'll assume p, Q, m are fixed and not fitted by curve_fit directly.
            # This fit method will primarily use the ODE solver with fixed p, Q, m.
            # The actual fitting of p, Q, m will be handled by a dedicated fitter class.
            
            # For Phase 1, this fit method will not actually optimize p, Q, m.
            # It will just demonstrate how the ODE is solved with given parameters.
            # The actual parameter fitting will be implemented in fitters/scipy_fitter.py
            # and will call this predict method internally.
            
            # For now, we'll just return the prediction based on the initialized p, Q, m
            # This method is primarily for demonstrating the ODE solution.
            
            # To make curve_fit work, we need to pass parameters to it.
            # Let's assume for now that we are fitting the initial conditions or some scaling factors
            # This part needs to be properly designed with a dedicated fitter.
            
            # For the purpose of making this class runnable in Phase 1, 
            # we'll make a dummy fit that just checks if the parameters are set.
            # The actual parameter estimation will be done by the fitter classes.
            
            # If we were to fit p, Q, m here, the signature would be complex.
            # For now, this fit method will be a placeholder that expects parameters to be set externally
            # or will be called by a fitter that manages the optimization.
            
            # Let's make a simple placeholder that just returns the current prediction
            # This will be refined when the fitter classes are implemented.
            
            # This `fit` method will be called by a `ScipyFitter` or `JaxFitter`.
            # The `ScipyFitter` will handle the `curve_fit` and pass the optimized parameters
            # back to this model instance.
            
            # So, for now, this `fit` method will not perform optimization itself.
            # It will be responsible for setting the internal parameters once they are optimized
            # by an external fitter.
            
            # To satisfy the abstract method requirement, we'll make it a no-op for now
            # and rely on the fitter to call predict and update params.
            
            # This is a temporary placeholder. The actual fitting logic will be in scipy_fitter.py
            # and will update self.p, self.Q, self.m after optimization.
            
            # For now, we'll just ensure the data is correctly formatted.
            pass

        # This method will be called by a fitter, which will handle the actual optimization
        # and then set the parameters of this model instance.
        # So, no optimization logic here for now.
        # We'll just store the data for potential use by a fitter.
        self._t_fit = t_arr
        self._y_fit_df = y_obs_df
        self._y_fit_flat = y_obs_flat

        # Placeholder for fitted parameters. These will be set by the fitter.
        # For demonstration, let's assume some dummy fitted parameters if not already set.
        if not self._fitted_params:
            self._fitted_params = {
                "p": self.p.tolist(),
                "Q": self.Q.tolist(),
                "m": self.m.tolist()
            }

        return self

    def predict(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self._fitted_params and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")
        
        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self._fitted_params.get("p", self.p))
        current_Q = B.array(self._fitted_params.get("Q", self.Q))
        current_m = B.array(self._fitted_params.get("m", self.m))

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

    def score(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self._fitted_params and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.")
        
        y_pred_df = self.predict(t)
        
        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(f"Observed data DataFrame must contain columns for all products: {self.names}")
        
        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, Union[float, List[float], List[List[float]]]]:
        # Return the parameters that were either initialized or fitted
        if self._fitted_params:
            return self._fitted_params
        else:
            return {"p": self.p.tolist(), "Q": self.Q.tolist(), "m": self.m.tolist()}

    def set_params(self, p: Sequence[float], Q: Sequence[Sequence[float]], m: Sequence[float]):
        """Manually set model parameters. Useful after external optimization."""
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self._fitted_params = {"p": self.p.tolist(), "Q": self.Q.tolist(), "m": self.m.tolist()}
