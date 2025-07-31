import numpy as np
import pandas as pd
import networkx as nx
import ndlib.models.ModelConfig as mc
from innovate.abm.ndlib_model import NDlibModel as NdlibInnovationModel
from innovate.preprocess import stl_decomposition
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.diffuse.logistic import LogisticModel
from innovate.utils.model_evaluation import get_fit_metrics


def test_ndlib_innovation_model_simulation():
    graph = nx.path_graph(5)
    model = NdlibInnovationModel(num_agents=5, graph=graph)
    config = mc.Configuration()
    for edge in graph.edges():
        config.add_edge_configuration("threshold", edge, 1.0)
    config.add_model_initial_configuration("Infected", [0])
    model.diffusion_model.set_initial_status(config)

    adoption_counts = []
    for _ in range(5):
        model.step()
        count = sum(1 for s in model.diffusion_model.status.values() if s in [1, 2])
        adoption_counts.append(count)

    assert adoption_counts == [1, 2, 3, 4, 5]


def test_stl_fit_and_score_pipeline():
    dates = pd.date_range(start="2020-01-01", periods=24, freq="M")
    true_model = LogisticModel()
    true_model.params_ = {"L": 100.0, "k": 1.5, "x0": 10.0}
    y = true_model.predict(np.arange(1, 25))
    series = pd.Series(y + np.random.normal(0, 5, len(y)), index=dates)

    decomposed = stl_decomposition(series, period=12)
    trend = decomposed["trend"]
    t = np.arange(1, len(trend) + 1)

    model = LogisticModel()
    fitter = ScipyFitter()
    fitter.fit(model, t, trend.values)

    metrics = get_fit_metrics(model, t, trend.values)
    assert "RMSE" in metrics and metrics["RMSE"] >= 0
