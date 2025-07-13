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

        This method is not implemented because the market share attraction model does not use instantaneous interaction rates.
        """
        # This model is not based on differential equations, so this method is not applicable.
        pass

    def predict_states(self, time_points, **params):
        """
        Predicts the states of the competing entities over time.

        Predicts the market share distribution of competing entities based on their relative attractiveness.
        
        Parameters:
            time_points: Ignored, as the model is not time-dependent.
            attractiveness (list): Attractiveness values for each competing entity.
        
        Returns:
            An array representing the normalized market shares for each entity, or a zero vector if total attractiveness is zero.
        
        Raises:
            ValueError: If attractiveness values are not provided.
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

        Return a schema describing the expected parameters for the market share attraction model.
        
        Returns:
            dict: A dictionary specifying that the model requires an "attractiveness" parameter, which is a list of values representing the attractiveness of each competing entity.
        """
        return {
            "attractiveness": {
                "type": "list",
                "default": [],
                "description": "A list of attractiveness values for each competing entity."
            }
        }
