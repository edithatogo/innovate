import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from innovate.compete.competition import MultiProductDiffusionModel
from innovate.fail.analysis import analyze_failure

def run_failed_adoption_example():
    print("--- Running Failed Adoption Example ---")

    # Define parameters for a multi-product model where one product is designed to fail
    # Product A: Moderate success
    # Product B: Low adoption, likely to fail
    # Product C: High success
    p_vals = [0.03, 0.005, 0.04]  # Intrinsic adoption rates
    Q_matrix = [
        [0.3, 0.05, 0.02],  # Q[0,0] = imitation for ProdA from ProdA, Q[0,1] = ProdA from ProdB, etc.
        [0.01, 0.1, 0.01],   # ProdB has low internal imitation and low influence from others
        [0.05, 0.02, 0.4]    # ProdC has high internal imitation
    ]
    m_vals = [1000, 200, 1200]  # Ultimate market potentials
    product_names = ["Product A", "Product B", "Product C"]

    model = MultiProductDiffusionModel(p=p_vals, Q=Q_matrix, m=m_vals, names=product_names)

    time_horizon = np.arange(1, 51) # 50 time points
    predictions_df = model.predict(time_horizon)

    print("\nCumulative Adoptions (last 5 time points):\n", predictions_df.tail())

    # Analyze for failed technologies
    # A product is considered failed if its maximum adoption is less than 10% of its potential
    # or a fixed threshold, let’s use a fixed threshold for simplicity here.
    # For market share, we need to normalize by total market potential or just use raw adoption.
    # The analyze_failure function expects market share, so let’s normalize predictions by their max potential.
    
    # Normalize predictions to represent market share (0 to 1)
    # This assumes m_vals are the total market potential for each product.
    # If predictions_df is cumulative adoptions, then max_adoption / m_val is the market share achieved.
    # Let’s adjust analyze_failure to work with absolute adoption if needed, or ensure input is market share.
    # For now, let’s assume analyze_failure works with values that can be compared to a threshold.
    # If predictions are cumulative adoptions, we can check if max adoption is below a threshold.
    
    # Let’s define failure based on absolute cumulative adoption not reaching a certain point.
    # For analyze_failure, it expects market share. Let’s convert predictions to market share.
    market_share_predictions = predictions_df.values / np.array(m_vals)

    failure_threshold = 0.1 # 10% of market potential
    failed_products_indices = analyze_failure(market_share_predictions, failure_threshold=failure_threshold)

    if failed_products_indices:
        print(f"\nIdentified Failed Products (threshold < {failure_threshold*100}% of market potential):")
        for idx in failed_products_indices:
            print(f"- {product_names[idx]} (Index: {idx})")
    else:
        print("\nNo products identified as failed based on the given threshold.")

    # Plotting the results
    plt.figure(figsize=(12, 7))
    for i, name in enumerate(product_names):
        plt.plot(time_horizon, predictions_df[name], label=f'{name} (m={m_vals[i]})')
        if i in failed_products_indices:
            plt.axhline(y=m_vals[i] * failure_threshold, color='r', linestyle='--', alpha=0.7, label=f'Failure Threshold ({name})' if i == failed_products_indices[0] else '')
            plt.text(time_horizon[-1] * 0.8, m_vals[i] * failure_threshold * 1.1, 'FAILED', color='red', fontsize=10, ha='center')

    plt.title('Multi-Product Diffusion with Potential Failures')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Adoptions')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_failed_adoption_example()
