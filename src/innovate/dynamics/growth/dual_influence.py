from .base import GrowthCurve
from innovate.backend import current_backend as B

class DualInfluenceGrowth(GrowthCurve):
    """
    Models growth driven by two distinct forces: external influence (innovators)
    and internal influence (imitators). The shape of the S-curve can be symmetric
    or asymmetric depending on the relative strength of these two forces. This is
    often referred to as the Bass model.

    Core Behavior: Captures the dynamics of products or ideas that are initially
    pushed by external sources (e.g., advertising) and then take off through
    word-of-mouth or social contagion.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Compute the instantaneous growth rate of adopters based on the Bass diffusion model.
        
        The growth rate is calculated as dN/dt = (p + q * (N/M)) * (M - N), where:
        - p: innovation coefficient (external influence)
        - q: imitation coefficient (internal influence)
        - N: current number of adopters
        - M: total potential adopters
        
        Parameters:
        	current_adopters (float): Current number of adopters.
        	total_potential (float): Total potential number of adopters.
        
        Returns:
        	float: The instantaneous growth rate. Returns 0 if total potential is not positive.
        """
        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential
        N = current_adopters
        return (p + q * (N / M)) * (M - N) if M > 0 else 0

    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts the cumulative number of adopters at specified time points using the Bass diffusion model.
        
        Parameters:
            time_points (array-like): Sequence of time points at which to predict cumulative adoption.
            initial_adopters (float): Number of adopters at the initial time point.
            total_potential (float): Total potential number of adopters.
        
        Returns:
            numpy.ndarray: Flattened array of cumulative adopters at each specified time point.
        """
        from scipy.integrate import solve_ivp

        p = params.get("innovation_coeff", 0.001)
        q = params.get("imitation_coeff", 0.1)
        M = total_potential

        fun = lambda t, y: self.compute_growth_rate(y, M, innovation_coeff=p, imitation_coeff=q)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            [initial_adopters],
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.flatten()

    def get_parameters_schema(self):
        """
        Return a schema describing the model parameters for innovation and imitation coefficients.
        
        Returns:
            dict: A dictionary specifying the type, default value, and description for each model parameter.
        """
        return {
            "innovation_coeff": {
                "type": "float",
                "default": 0.001,
                "description": "The coefficient of innovation (external influence)."
            },
            "imitation_coeff": {
                "type": "float",
                "default": 0.1,
                "description": "The coefficient of imitation (internal influence)."
            }
        }
