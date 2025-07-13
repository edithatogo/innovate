from .base import SystemBehavior
from innovate.backend import current_backend as B

class HypeCycleBehavior(SystemBehavior):
    """
    Models the rise and fall of expectations through coupled expectation and
    maturity stocks.
    """

    def compute_behavior_rates(self, **params):
        """
        Calculates the instantaneous behavior rates.

        Equations:
        dE/dt = alpha1 * Innovation_Trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dM/dt = beta1 * R&D_Investment(E) + beta2 * M - beta3 * M

        Compute the instantaneous rates of change for expectation and maturity stocks based on model parameters.
        
        Calculates the derivatives dE/dt and dM/dt using coupled differential equations, incorporating effects from innovation triggers, interaction coefficients, and R&D investment as a function of expectation.
        
        Returns:
            dEdt (float): Rate of change of expectation.
            dMdt (float): Rate of change of maturity.
        """
        E = params.get("E")
        M = params.get("M")

        # Parameters for the Expectation equation
        alpha1 = params.get("alpha1", 0.1)
        alpha2 = params.get("alpha2", 0.01)
        alpha3 = params.get("alpha3", 0.05)
        alpha4 = params.get("alpha4", 0.001)
        innovation_trigger = params.get("innovation_trigger", 0)

        # Parameters for the Maturity equation
        beta1 = params.get("beta1", 0.01)
        beta2 = params.get("beta2", 0.02)
        beta3 = params.get("beta3", 0.01)

        # R&D investment is a function of expectations
        rd_investment = params.get("rd_investment_factor", 0.1) * E

        dEdt = alpha1 * innovation_trigger + alpha2 * M * E - alpha3 * E + alpha4 * (E - M) * E
        dMdt = beta1 * rd_investment + beta2 * M - beta3 * M

        return dEdt, dMdt

    def predict_states(self, time_points, **params):
        """
        Predicts the states of the system over time.

        Simulate the evolution of expectation and maturity states over specified time points.
        
        Parameters:
            time_points (array-like): Sequence of time points at which to evaluate the system states.
        
        Returns:
            ndarray: Array of shape (len(time_points), 2) containing the predicted expectation and maturity values at each time point.
        """
        from scipy.integrate import solve_ivp

        E0 = params.get("E0", 1)
        M0 = params.get("M0", 1)

        fun = lambda t, y: self.compute_behavior_rates(E=y[0], M=y[1], **params)

        sol = solve_ivp(
            fun,
            (time_points[0], time_points[-1]),
            [E0, M0],
            t_eval=time_points,
            method='LSODA',
        )
        return sol.y.T

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.

        Return a dictionary schema describing all model parameters, including their types and default values.
        
        Returns:
            dict: A mapping of parameter names to their type and default value for the hype cycle model.
        """
        return {
            "alpha1": {"type": "float", "default": 0.1},
            "alpha2": {"type": "float", "default": 0.01},
            "alpha3": {"type": "float", "default": 0.05},
            "alpha4": {"type": "float", "default": 0.001},
            "innovation_trigger": {"type": "float", "default": 0},
            "beta1": {"type": "float", "default": 0.01},
            "beta2": {"type": "float", "default": 0.02},
            "beta3": {"type": "float", "default": 0.01},
            "rd_investment_factor": {"type": "float", "default": 0.1},
            "E0": {"type": "float", "default": 1},
            "M0": {"type": "float", "default": 1}
        }
