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
        Calculates the instantaneous growth rate.
        """
        # The differential form of the Gompertz model is more complex
        # and less intuitive than the cumulative form. For simplicity,
        # we will use the cumulative form to calculate the rate.
        # This is not ideal, but it is a reasonable approximation.
        t = params.get("t")
        if t is None:
            raise ValueError("SkewedGrowth requires time points to be provided as a parameter.")

        y_pred = self.predict_cumulative(t, current_adopters, total_potential, **params)

        if y_pred.ndim == 0:
            y_pred = B.array([y_pred])

        t = B.array(t)
        if t.ndim == 0:
            t = B.array([t])

        # Calculate the rate as the difference between consecutive points
        rate = B.diff(y_pred) / B.diff(t)

        # Return the last calculated rate for the current time point
        return rate[-1] if len(rate) > 0 else 0


    def predict_cumulative(self, time_points, initial_adopters, total_potential, **params):
        """
        Predicts cumulative adopters over time.

        Equation: N(t) = K * exp(-b * exp(-c*t))
        """
        K = total_potential
        b = params.get("shape_b", 1.0)
        c = params.get("shape_c", 0.1)

        return K * B.exp(-b * B.exp(-c * B.array(time_points)))

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.
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
