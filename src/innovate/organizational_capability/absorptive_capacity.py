from innovate.base.base import DiffusionModel

class AbsorptiveCapacity(DiffusionModel):
    """
    This model explicitly tracks a firm's ability to absorb and utilize
    external knowledge, which in turn influences its effective knowledge
    acquisition.
    """

    def __init__(self, **params):
        self._params = params

    @property
    def param_names(self):
        return ["ac_build_from_rd", "ac_build_from_collaboration", "ac_decay_rate", "initial_absorptive_capacity"]

    def initial_guesses(self, t, y):
        return {
            "ac_build_from_rd": 0.1,
            "ac_build_from_collaboration": 0.05,
            "ac_decay_rate": 0.02,
            "initial_absorptive_capacity": 1.0,
        }

    def bounds(self, t, y):
        return {
            "ac_build_from_rd": (0, 1),
            "ac_build_from_collaboration": (0, 1),
            "ac_decay_rate": (0, 1),
            "initial_absorptive_capacity": (0, None),
        }

    def differential_equation(self, y, t, p):
        """
        The differential equation for the Absorptive Capacity model.
        """
        internal_rd_investment = p.get("internal_rd_investment", 0)
        collaboration_effort = p.get("collaboration_effort", 0)

        ac_building_rate = (
            self._params["ac_build_from_rd"] * internal_rd_investment +
            self._params["ac_build_from_collaboration"] * collaboration_effort
        )

        ac_erosion_rate = self._params["ac_decay_rate"] * y[0]

        return ac_building_rate - ac_erosion_rate

    def predict(self, t):
        from scipy.integrate import solve_ivp

        y0 = [self._params.get("initial_absorptive_capacity", 1.0)]

        sol = solve_ivp(
            self.differential_equation,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=(self._params,),
        )
        return sol.y[0]
