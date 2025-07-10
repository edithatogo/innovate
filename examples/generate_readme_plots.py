"""
This script generates a gallery of example plots to be used in the README.md.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

from innovate.diffuse.bass import BassModel
from innovate.compete.lotka_volterra import LotkaVolterraModel
from innovate.hype.hype_cycle import HypeCycleModel
from innovate.reduce.analysis import identify_reducing_series, smooth_series

# --- Configuration ---
SAVE_DIR = "docs/images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Bass Diffusion Curve ---
print("Generating Bass Diffusion plot...")
bass_model = BassModel()
t = np.linspace(0, 20, 100)
p, q, m = 0.03, 0.38, 1000
y_bass = bass_model.cumulative_adoption(t, p, q, m)

plt.figure(figsize=(8, 5))
plt.plot(t, y_bass, label=f'p={p}, q={q}, m={m}')
plt.title("Bass Diffusion Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "bass_diffusion.png"))
plt.close()

# --- 2. Lotka-Volterra Competition ---
print("Generating Lotka-Volterra Competition plot...")
lv_model = LotkaVolterraModel()
lv_model.params_ = {"alpha1": 0.6, "beta1": 0.1, "alpha2": 0.4, "beta2": 0.1}
t_lv = np.arange(0, 40, 1)
y0_lv = [0.01, 0.02]
y_lv = lv_model.predict(t_lv, y0_lv)

plt.figure(figsize=(8, 5))
plt.plot(t_lv, y_lv[:, 0], label='Product 1')
plt.plot(t_lv, y_lv[:, 1], label='Product 2')
plt.title("Lotka-Volterra Competition")
plt.xlabel("Time")
plt.ylabel("Market Share")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lotka_volterra_competition.png"))
plt.close()

# --- 3. Hype Cycle ---
print("Generating Hype Cycle plot...")
hype_model = HypeCycleModel()
# These parameters are for demonstration and create a classic hype shape
hype_model.params_ = {
    "k": 0.1, "t0": 20, "a_hype": 1.0, "t_hype": 4, "w_hype": 2,
    "a_d": 0.6, "t_d": 8, "w_d": 4
}
t_hype = np.linspace(0, 30, 200)
y_hype = hype_model.predict(t_hype)

plt.figure(figsize=(8, 5))
plt.plot(t_hype, y_hype)
plt.title("Gartner Hype Cycle Model")
plt.xlabel("Time")
plt.ylabel("Expectations / Visibility")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hype_cycle.png"))
plt.close()

# --- 4. Reduction Analysis ---
print("Generating Reduction Analysis plot...")
# Create a synthetic series that rises and then falls
t_reduce = np.arange(50)
reducing_series = np.concatenate([
    np.linspace(10, 80, 25),
    np.linspace(80, 30, 25)
]) + np.random.normal(0, 5, 50)

# Run the analysis
analysis_df = identify_reducing_series([reducing_series])
changepoint = int(analysis_df.loc[0, 'changepoint_index'])
smoothed_series = smooth_series(reducing_series, fraction=0.2)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t_reduce, reducing_series, 'o', alpha=0.5, label='Raw Data')
plt.plot(t_reduce, smoothed_series, '-', linewidth=2, label='Smoothed Trend')
if changepoint != -1:
    plt.axvline(changepoint, color='r', linestyle='--', label=f'Detected Changepoint (t={changepoint})')
plt.title("Reduction Analysis")
plt.xlabel("Time")
plt.ylabel("Utilisation")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "reduction_analysis.png"))
plt.close()

print("All plots generated successfully.")
