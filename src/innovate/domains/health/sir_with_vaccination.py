from innovate.dynamics.contagion import SIRModel

class SIRWithVaccination(SIRModel):
    """
    A simple SIR model with vaccination.
    """

    def compute_spread_rate(self, **params):
        """
        Calculates the instantaneous spread rate.
        """
        S = params.get("S")
        I = params.get("I")
        beta = params.get("transmission_rate", 0.1)
        gamma = params.get("recovery_rate", 0.01)
        vaccination_rate = params.get("vaccination_rate", 0.0)

        dSdt = -beta * S * I - vaccination_rate * S
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I + vaccination_rate * S
        return dSdt, dIdt, dRdt

    def get_parameters_schema(self):
        """
        Returns the schema for the model's parameters.
        """
        schema = super().get_parameters_schema()
        schema["vaccination_rate"] = {
            "type": "float",
            "default": 0.0,
            "description": "The rate of vaccination."
        }
        return schema
