from innovate.base.base import DiffusionModel

class PlatformMultiSidedGrowth(DiffusionModel):
    """
    Models how the growth on one side of a platform (e.g., app developers)
    influences the growth on the other side (e.g., users), creating strong
    indirect network effects.
    """

    def __init__(self, **params):
        self._params = params

    @property
    def param_names(self):
        return [
            "alpha_users", "beta_users", "m_users", "churn_users",
            "alpha_devs", "beta_devs", "m_devs", "churn_devs",
            "initial_users", "initial_devs"
        ]

    def initial_guesses(self, t, y):
        return {
            "alpha_users": 0.1,
            "beta_users": 0.01,
            "m_users": 1000,
            "churn_users": 0.01,
            "alpha_devs": 0.05,
            "beta_devs": 0.02,
            "m_devs": 100,
            "churn_devs": 0.02,
            "initial_users": 1.0,
            "initial_devs": 1.0,
        }

    def bounds(self, t, y):
        return {
            "alpha_users": (0, 1),
            "beta_users": (0, 1),
            "m_users": (0, None),
            "churn_users": (0, 1),
            "alpha_devs": (0, 1),
            "beta_devs": (0, 1),
            "m_devs": (0, None),
            "churn_devs": (0, 1),
            "initial_users": (0, None),
            "initial_devs": (0, None),
        }

    def differential_equation(self, y, t, p):
        """
        The differential equation for the Platform Multi-Sided Growth model.
        """
        users, devs = y

        # User growth
        d_users_dt = (
            self._params["alpha_users"] + self._params["beta_users"] * devs
        ) * (self._params["m_users"] - users) - self._params["churn_users"] * users

        # Developer growth
        d_devs_dt = (
            self._params["alpha_devs"] + self._params["beta_devs"] * users
        ) * (self._params["m_devs"] - devs) - self._params["churn_devs"] * devs

        return [d_users_dt, d_devs_dt]

    def predict(self, t):
        from scipy.integrate import solve_ivp

        y0 = [
            self._params.get("initial_users", 1.0),
            self._params.get("initial_devs", 1.0)
        ]

        sol = solve_ivp(
            self.differential_equation,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=(self._params,),
        )
        return sol.y.T
