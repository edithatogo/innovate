import panel as pn
import numpy as np
from innovate.diffuse import BassModel

def create_bass_dashboard():
    """
    Creates an interactive dashboard for the Bass model.
    """
    pn.extension()

    # Create the widgets
    p = pn.widgets.FloatSlider(name="Innovation Coefficient (p)", start=0, end=0.1, step=0.001, value=0.001)
    q = pn.widgets.FloatSlider(name="Imitation Coefficient (q)", start=0, end=1.0, step=0.01, value=0.1)
    m = pn.widgets.FloatSlider(name="Market Potential (m)", start=100, end=10000, step=100, value=1000)
    t_max = pn.widgets.IntSlider(name="Time (t)", start=10, end=1000, step=10, value=100)

    # Create the plot
    @pn.depends(p, q, m, t_max)
    def plot_bass_model(p, q, m, t_max):
        t = np.linspace(0, t_max, t_max + 1)
        model = BassModel()
        model.params_ = {"p": p, "q": q, "m": m}
        y = model.predict(t)
        return pn.Column(
            pn.pane.Markdown("## Bass Diffusion Model"),
            pn.Row(
                pn.Column(p, q, m, t_max),
                pn.pane.Matplotlib(get_plot(t, y)),
            ),
        )

    def get_plot(t, y):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(t, y)
        plt.xlabel("Time")
        plt.ylabel("Adoptions")
        plt.title("Bass Diffusion Curve")
        return fig

    return plot_bass_model

if __name__ == "__main__":
    dashboard = create_bass_dashboard()
    dashboard.servable()
