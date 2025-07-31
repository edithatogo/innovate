import numpy as np
from innovate.diffuse.logistic import LogisticModel
from innovate.adopt.categorization import categorize_adopters
from innovate.fitters.scipy_fitter import ScipyFitter


def test_categorize_adopters():
    model = LogisticModel()
    # Generate some dummy data for fitting, as categorize_adopters needs a fitted model
    t_dummy = np.linspace(0, 20, 100)
    # Use the true parameters to generate data for fitting
    y_dummy = model.cumulative_adoption(t_dummy, L=1000, k=0.5, x0=10)

    fitter = ScipyFitter()
    fitter.fit(model, t_dummy, y_dummy)  # Fit the model

    t = np.linspace(0, 20, 100)  # Use the original t for categorization

    categorization_df = categorize_adopters(model, t)

    assert categorization_df is not None
    assert "category" in categorization_df.columns
    assert len(categorization_df) == len(t)

    # Check that all categories are present
    assert "Innovators" in categorization_df["category"].values
    assert "Early Adopters" in categorization_df["category"].values
    assert "Early Majority" in categorization_df["category"].values
    assert "Late Majority" in categorization_df["category"].values
    assert "Laggards" in categorization_df["category"].values
