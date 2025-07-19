from .base import GrowthCurve
from innovate.backend import current_backend as B

class SkewedGrowth(GrowthCurve):
    """
    Models asymmetric S-shaped growth where the rate of adoption is not
    symmetric around the inflection point. The inflection point is typically
    earlier than 50% of the market potential (around 37%), leading to a
    growth phase that decelerates more slowly than it accelerates. This is
    often referred to as the Gompertz growth model.

    Core Behavior: Represents growth with diminishing returns to scale or
    a rapid initial uptake followed by a long tail of adoption.
    """

    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Calculates the instantaneous growth rate using the Gompertz differential equation.

        Equation: dN/dt = c * N * (log(K) - log(N))
        """
        K = total_potential
        N = current_adopters[0] if hasattr(current_adopters, '__len__') else current_adopters
        c = params.get("shape_c", 0.1)

        if K <= 0 or N <= 0:
            return 0
        
        # The 'b' parameter is part of the integrated form, not the differential equation.
        # The rate is determined by the ceiling K, current level N, and growth rate c.
        return c * N * (B.log(K) - B.log(N))

    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts cumulative adopters over time.
        
        Equation: N(t) = K * exp(-b * exp(-c*t))

        Predict the cumulative number of adopters at specified time points using the Gompertz growth model.
        
        Parameters:
            time_points: Sequence of time values at which to predict cumulative adoption.
            initial_adopters: Initial number of adopters (not used in the Gompertz calculation but included for interface consistency).
            total_potential: The carrying capacity or total market potential.
            **params: Optional model parameters:
                - shape_b (float): Shape parameter controlling the displacement along the time axis (default: 1.0).
                - shape_c (float): Shape parameter controlling the growth rate (default: 0.1).
        
        Returns:
            Predicted cumulative adopters at each time point as an array.
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.

        Return a dictionary describing the schema for the Gompertz model parameters `shape_b` and `shape_c`.
        
        Returns:
            dict: Parameter schema including type, default value, and description for each model parameter.
        """
        return {
            "shape_b": {
                "type": "float",
                "default": 1.0,
                "description": "Shape parameter b."
            },
            "shape_c": {
                "type": "float",
                "default": 0.1,
                "description": "Shape parameter c."
            }
        }
