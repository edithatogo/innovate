from .base import CompetitiveInteraction
from innovate.backend import current_backend as B

class MarketShareAttraction(CompetitiveInteraction):
    """
    Determines market share based on relative attractiveness, which can be
    dynamically influenced by attributes (e.g., price, quality).
    """

    def compute_interaction_rates(self, **params):
        """
        Calculates the instantaneous interaction rates.
        """
        # This model is not based on differential equations, so this method is not applicable.
        pass

    def predict_states(self, time_points, **params):
        """
        Predicts the states of the competing entities over time.
        """
        # This model is not time-dependent in the same way as the other models.
        # It calculates the market share at a single point in time based on the
        # attractiveness of the competing entities.

        attractiveness = params.get("attractiveness", [])
        if not attractiveness:
            raise ValueError("Attractiveness values must be provided.")

        total_attractiveness = B.sum(B.array(attractiveness))

        if total_attractiveness == 0:
            return B.zeros(len(attractiveness))

        return B.array(attractiveness) / total_attractiveness

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity."
            }
        }
