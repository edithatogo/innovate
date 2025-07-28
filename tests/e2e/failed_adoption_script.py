import numpy as np
import pandas as pd

# Simplified version of the failed adoption example

def run_failed_adoption_example():
    """Runs a simplified simulation of product adoption to identify failed products.

    A product is considered to have failed if its market share never exceeds
    a predefined threshold.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Adoption predictions over time.
            - list: Indices of products identified as failed.
    """
    # Adoption parameters for three products
    # Use a much smaller adoption rate for Product B so it fails
    p_vals = np.array([0.03, 0.002, 0.04])
    m_vals = np.array([1000, 200, 1200])
    product_names = ["Product A", "Product B", "Product C"]

    # Time horizon (50 periods)
    t = np.arange(1, 51)

    # Simple exponential adoption curve (no competition)
    data = {}
    for p, m, name in zip(p_vals, m_vals, product_names):
        data[name] = m * (1 - np.exp(-p * t))
    predictions_df = pd.DataFrame(data, index=t)

    # Identify failed products based on market share threshold
    market_share = predictions_df.values / m_vals
    failure_threshold = 0.1
    failed = list(np.where(market_share.max(axis=0) < failure_threshold)[0])
    return predictions_df, failed

if __name__ == "__main__":
    df, failed = run_failed_adoption_example()
    print(df.tail())
    print(f"Failed indices: {failed}")
