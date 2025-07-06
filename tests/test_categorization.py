import pytest
import numpy as np
from heartflow.models.logistic import LogisticModel
from heartflow.utils.categorization import categorize_adopters

def test_categorize_adopters():
    model = LogisticModel()
    model._L = 1000
    model._k = 0.5
    model._x0 = 10

    t = np.linspace(0, 20, 100)
    
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
