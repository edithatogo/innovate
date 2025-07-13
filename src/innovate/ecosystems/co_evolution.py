from innovate.base.base import DiffusionModel

class CoEvolution(DiffusionModel):
    """
    Models the mutually reinforcing (or inhibiting) development of an
    innovation and its critical supporting elements (e.g., electric vehicles
    and charging stations).
    """

    def __init__(self, **params):
        self._params = params

    @property
    def param_names(self):
        return [
            "p_tech", "q_tech", "m_tech", "k_half",
            "investment_rate", "depreciation_rate",
            "initial_tech_adopters", "initial_infrastructure"
        ]

    def initial_guesses(self, t, y):
        return {
            "p_tech": 0.001,
            "q_tech": 0.1,
            "m_tech": 1000,
            "k_half": 500,
            "investment_rate": 0.1,
            "depreciation_rate": 0.01,
            "initial_tech_adopters": 1.0,
            "initial_infrastructure": 1.0,
        }

    def bounds(self, t, y):
        return {
            "p_tech": (0, 1),
            "q_tech": (0, 1),
            "m_tech": (0, None),
            "k_half": (0, None),
            "investment_rate": (0, 1),
            "depreciation_rate": (0, 1),
            "initial_tech_adopters": (0, None),
            "initial_infrastructure": (0, None),
        }

    def differential_equation(self, y, t, p):
        """
        The differential equation for the Co-Evolution model.
        """
        tech_adopters, infrastructure = y

        # Technology adoption rate
        f_I = infrastructure / (infrastructure + self._params["k_half"])
        d_tech_adopters_dt = (
            self._params["p_tech"] * (self._params["m_tech"] - tech_adopters) +
            self._params["q_tech"] * (tech_adopters / self._params["m_tech"]) * (self._params["m_tech"] - tech_adopters)
        ) * f_I

        # Infrastructure development rate
        g_T = tech_adopters / self._params["m_tech"]
        d_infrastructure_dt = (
            self._params["investment_rate"] * g_T -
            self._params["depreciation_rate"] * infrastructure
        )

        return [d_tech_adopters_dt, d_infrastructure_dt]

    def predict(self, t):
        from scipy.integrate import solve_ivp

        y0 = [
            self._params.get("initial_tech_adopters", 1.0),
            self._params.get("initial_infrastructure", 1.0)
        ]

        sol = solve_ivp(
            self.differential_equation,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=(self._params,),
        )
        return sol.y.T
