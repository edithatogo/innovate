from innovate.base.base import DiffusionModel

class KnowledgeAccumulation(DiffusionModel):
    """
    Models the accumulation and depreciation of an organization's internal
    knowledge base, which fuels its innovation output.
    """

    def __init__(self, **params):
        self._params = params

    @property
    def param_names(self):
        return ["rd_effectiveness", "learning_by_doing_rate", "external_learning_coeff", "knowledge_depreciation_rate", "turnover_sensitivity", "innovation_elasticity", "initial_knowledge"]

    def initial_guesses(self, t, y):
        return {
            "rd_effectiveness": 0.1,
            "learning_by_doing_rate": 0.01,
            "external_learning_coeff": 0.05,
            "knowledge_depreciation_rate": 0.02,
            "turnover_sensitivity": 0.1,
            "innovation_elasticity": 1.0,
            "initial_knowledge": 1.0,
        }

    def bounds(self, t, y):
        return {
            "rd_effectiveness": (0, 1),
            "learning_by_doing_rate": (0, 1),
            "external_learning_coeff": (0, 1),
            "knowledge_depreciation_rate": (0, 1),
            "turnover_sensitivity": (0, 1),
            "innovation_elasticity": (0, 2),
            "initial_knowledge": (0, None),
        }

    def differential_equation(self, y, t, p):
        """
        The differential equation for the Knowledge Accumulation model.
        """
        rd_investment = p.get("rd_investment", 0)
        experience = p.get("experience", 0)
        external_learning = p.get("external_learning", 0)
        turnover_rate = p.get("turnover_rate", 0)

        knowledge_acquisition_rate = (
            self._params["rd_effectiveness"] * rd_investment +
            self._params["learning_by_doing_rate"] * experience +
            self._params["external_learning_coeff"] * external_learning
        )

        knowledge_depreciation_rate = (
            self._params["knowledge_depreciation_rate"] * y[0] +
            self._params["turnover_sensitivity"] * turnover_rate * y[0]
        )

        return knowledge_acquisition_rate - knowledge_depreciation_rate

    def predict(self, t):
        from scipy.integrate import solve_ivp

        y0 = [self._params.get("initial_knowledge", 1.0)]

        sol = solve_ivp(
            self.differential_equation,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=(self._params,),
        )
        return sol.y[0]
