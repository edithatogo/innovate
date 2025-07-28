import pandas as pd
from innovate.abm.model import InnovationModel
from innovate.abm.disruptive_innovation import DisruptiveInnovationModel
from innovate.abm.sentiment_hype_cycle import SentimentHypeModel
from innovate.abm.competitive_diffusion import CompetitiveDiffusionModel


def test_innovation_model_basic():
    model = InnovationModel(num_agents=5, width=5, height=5)
    assert len(model.agents) == 5
    assert all(agent.pos is not None for agent in model.agents)
    model.step()
    assert model.steps == 1


def test_disruptive_innovation_model_run():
    model = DisruptiveInnovationModel(
        num_agents=5,
        width=5,
        height=5,
        initial_disruptive_performance=0.5,
        disruptive_performance_improvement=0.1,
    )
    df = model.run_model(3)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["IncumbentAdopters", "DisruptiveAdopters"]
    assert len(df) == 3


def test_sentiment_hype_model_run():
    model = SentimentHypeModel(
        num_agents=5,
        width=5,
        height=5,
        adoption_threshold=1,
        sentiment_threshold=1,
    )
    df = model.run_model(2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Adopters"]
    assert len(df) == 2


def test_competitive_diffusion_model_run():
    model = CompetitiveDiffusionModel(
        num_agents=5,
        width=5,
        height=5,
        num_innovations=2,
    )
    df = model.run_model(2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["AdoptionCounts"]
    assert len(df) == 2
    for counts in df["AdoptionCounts"]:
        assert len(counts) == 2
