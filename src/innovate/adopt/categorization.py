from innovate.backend import current_backend as B
import pandas as pd
from innovate.base.base import DiffusionModel
from typing import Sequence

def categorize_adopters(model: DiffusionModel, t: Sequence[float]) -> pd.DataFrame:
    """:no-index:

    Categorizes adopters based on the fitted diffusion model.

    Args:
        model: A fitted diffusion model.
        t: A sequence of time points.

    Returns:
        A pandas DataFrame with the adopter categories for each time point.
    """
    adoption_rate = model.predict_adoption_rate(t)
    
    # Calculate mean and standard deviation of the adoption rate
    mean_adoption_time = B.sum(t * adoption_rate) / B.sum(adoption_rate)
    std_dev_adoption_time = B.sqrt(B.sum(((t - mean_adoption_time) ** 2) * adoption_rate) / B.sum(adoption_rate))
    
    # Define category boundaries
    innovators_end = mean_adoption_time - 2 * std_dev_adoption_time
    early_adopters_end = mean_adoption_time - std_dev_adoption_time
    early_majority_end = mean_adoption_time
    late_majority_end = mean_adoption_time + std_dev_adoption_time
    
    # Categorize each time point
    categories = []
    for time_point in t:
        if time_point <= innovators_end:
            categories.append("Innovators")
        elif time_point <= early_adopters_end:
            categories.append("Early Adopters")
        elif time_point <= early_majority_end:
            categories.append("Early Majority")
        elif time_point <= late_majority_end:
            categories.append("Late Majority")
        else:
            categories.append("Laggards")
            
    return pd.DataFrame({"time": t, "adoption_rate": adoption_rate, "category": categories})