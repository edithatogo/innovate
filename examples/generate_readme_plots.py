"""
This script generates a gallery of example plots to be used in the README.md.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

from innovate.diffuse.bass import BassModel
from innovate.compete.lotka_volterra import LotkaVolterraModel
from innovate.hype.hype_cycle import HypeCycleModel
from innovate.reduce.analysis import identify_reducing_series, smooth_series
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.substitute.fisher_pry import FisherPryModel
from innovate.substitute.norton_bass import NortonBassModel
from innovate.compete.multi_product import MultiProductDiffusionModel

# --- Configuration ---
SAVE_DIR = "docs/images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Bass Diffusion Curve ---
print("Generating Bass Diffusion plot...")
bass_model = BassModel()
t = np.linspace(0, 20, 100)
p, q, m = 0.03, 0.38, 1000
bass_model.params_ = {"p": p, "q": q, "m": m}
y_bass = bass_model.predict(t)

plt.figure(figsize=(8, 5))
plt.plot(t, y_bass, label=f"p={p}, q={q}, m={m}")
plt.title("Bass Diffusion Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle="--", alpha=0.6)
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
plt.plot(t_lv, y_lv[:, 0], label="Product 1")
plt.plot(t_lv, y_lv[:, 1], label="Product 2")
plt.title("Lotka-Volterra Competition")
plt.xlabel("Time")
plt.ylabel("Market Share")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "lotka_volterra_competition.png"))
plt.close()

# --- 3. Hype Cycle ---
print("Generating Hype Cycle plot...")
hype_model = HypeCycleModel()
# These parameters are for demonstration and create a classic hype shape
hype_model.params_ = {
    "k": 0.1,
    "t0": 20,
    "a_hype": 1.0,
    "t_hype": 4,
    "w_hype": 2,
    "a_d": 0.6,
    "t_d": 8,
    "w_d": 4,
}
t_hype = np.linspace(0, 30, 200)
y_hype = hype_model.predict(t_hype)

plt.figure(figsize=(8, 5))
plt.plot(t_hype, y_hype)
plt.title("Gartner Hype Cycle Model")
plt.xlabel("Time")
plt.ylabel("Expectations / Visibility")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hype_cycle.png"))
plt.close()

# --- 4. Reduction Analysis ---
print("Generating Reduction Analysis plot...")
# Create a synthetic series that rises and then falls
t_reduce = np.arange(50)
reducing_series = np.concatenate(
    [np.linspace(10, 80, 25), np.linspace(80, 30, 25)]
) + np.random.normal(0, 5, 50)

# Run the analysis
analysis_df = identify_reducing_series([reducing_series])
changepoint = int(analysis_df.loc[0, "changepoint_index"])
smoothed_series = smooth_series(reducing_series, fraction=0.2)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t_reduce, reducing_series, "o", alpha=0.5, label="Raw Data")
plt.plot(t_reduce, smoothed_series, "-", linewidth=2, label="Smoothed Trend")
if changepoint != -1:
    plt.axvline(
        changepoint,
        color="r",
        linestyle="--",
        label=f"Detected Changepoint (t={changepoint})",
    )
plt.title("Reduction Analysis")
plt.xlabel("Time")
plt.ylabel("Utilisation")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "reduction_analysis.png"))
plt.close()

# --- 5. Gompertz Diffusion Curve ---
print("Generating Gompertz Diffusion plot...")
gompertz_model = GompertzModel()
gompertz_model.params_ = {"a": 1000, "b": 0.1, "c": 0.1}
t_gompertz = np.linspace(0, 50, 100)
y_gompertz = gompertz_model.predict(t_gompertz)

plt.figure(figsize=(8, 5))
plt.plot(t_gompertz, y_gompertz, label="a=1000, b=0.1, c=0.1")
plt.title("Gompertz Diffusion Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "gompertz_diffusion.png"))
plt.close()

# --- 6. Logistic Diffusion Curve ---
print("Generating Logistic Diffusion plot...")
logistic_model = LogisticModel()
logistic_model.params_ = {"L": 1000, "k": 0.2, "x0": 25}
t_logistic = np.linspace(0, 50, 100)
y_logistic = logistic_model.predict(t_logistic)

plt.figure(figsize=(8, 5))
plt.plot(t_logistic, y_logistic, label="L=1000, k=0.2, x0=25")
plt.title("Logistic Diffusion Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "logistic_diffusion.png"))
plt.close()

# --- 7. Fisher-Pry Substitution ---
print("Generating Fisher-Pry Substitution plot...")
fisher_pry_model = FisherPryModel()
fisher_pry_model.params_ = {"alpha": 0.1, "t0": 0}
t_fisher_pry = np.linspace(-20, 20, 100)
y_fisher_pry = fisher_pry_model.predict(t_fisher_pry)

plt.figure(figsize=(8, 5))
plt.plot(t_fisher_pry, y_fisher_pry, label="alpha=0.1, beta=0.2")
plt.title("Fisher-Pry Substitution Model")
plt.xlabel("Time")
plt.ylabel("Market Share of New Technology")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "fisher_pry_substitution.png"))
plt.close()

# --- 8. Norton-Bass Substitution ---
print("Generating Norton-Bass Substitution plot...")
norton_bass_model = NortonBassModel(n_generations=2)
norton_bass_model.params_ = {
    "p1": 0.03,
    "q1": 0.2,
    "m1": 1000,
    "p2": 0.02,
    "q2": 0.3,
    "m2": 1500,
}
t_norton_bass = np.linspace(0, 50, 100)
y_norton_bass = norton_bass_model.predict(t_norton_bass)

plt.figure(figsize=(8, 5))
plt.plot(t_norton_bass, y_norton_bass[:, 0], label="Generation 1")
plt.plot(t_norton_bass, y_norton_bass[:, 1], label="Generation 2")
plt.title("Norton-Bass Substitution Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "norton_bass_substitution.png"))
plt.close()

# --- 9. Multi-Product Diffusion ---
print("Generating Multi-Product Diffusion plot...")
multi_product_model = MultiProductDiffusionModel(n_products=2)
multi_product_model.params_ = {
    "p1": 0.03,
    "p2": 0.02,
    "q1": 0.1,
    "q2": 0.15,
    "m1": 1000,
    "m2": 1200,
    "alpha_1_2": 0.5,
    "alpha_2_1": 0.3,
}
t_multi_product = np.linspace(0, 50, 100)
y_multi_product = multi_product_model.predict(t_multi_product)

plt.figure(figsize=(8, 5))
plt.plot(t_multi_product, y_multi_product[:, 0], label="Product 1")
plt.plot(t_multi_product, y_multi_product[:, 1], label="Product 2")
plt.title("Multi-Product Diffusion Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Adopters")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "multi_product_diffusion.png"))
plt.close()

# --- 10. Adoption Curve with Categories ---
print("Generating Adoption Curve plot with categories...")
# Use the Bass model results from before
adoption_rate = np.diff(y_bass, prepend=0)

# Find the peak adoption time (mean) and standard deviation
peak_time_idx = np.argmax(adoption_rate)
peak_time = t[peak_time_idx]

# Approximate standard deviation assuming the curve is roughly normal
# This is a simplification for visualization purposes
cumulative_adoption = y_bass / m
# Find time to 16% and 84% adoption
t_16 = t[np.argmin(np.abs(cumulative_adoption - 0.16))]
t_84 = t[np.argmin(np.abs(cumulative_adoption - 0.84))]
std_dev_approx = (t_84 - t_16) / 2

# Define category boundaries based on standard deviations from the peak
innovators_end = peak_time - 2 * std_dev_approx
early_adopters_end = peak_time - 1 * std_dev_approx
early_majority_end = peak_time
late_majority_end = peak_time + 1 * std_dev_approx

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Cumulative Adopters
ax1.plot(t, y_bass, "k--", label="Cumulative Adopters", alpha=0.7)
ax1.set_xlabel("Time")
ax1.set_ylabel("Cumulative Adopters")
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_ylim(0, m * 1.1)

# Plot Adoption Rate on the same y-axis for shading
ax2 = ax1
ax2.plot(t, adoption_rate, "k-", label="Adoption Rate")
ax2.set_ylabel("Adopters per Period")

# Shade the adopter categories
ax2.fill_between(
    t,
    adoption_rate,
    where=(t <= innovators_end),
    color="skyblue",
    alpha=0.6,
    label="Innovators (2.5%)",
)
ax2.fill_between(
    t,
    adoption_rate,
    where=(t > innovators_end) & (t <= early_adopters_end),
    color="lightgreen",
    alpha=0.6,
    label="Early Adopters (13.5%)",
)
ax2.fill_between(
    t,
    adoption_rate,
    where=(t > early_adopters_end) & (t <= early_majority_end),
    color="gold",
    alpha=0.6,
    label="Early Majority (34%)",
)
ax2.fill_between(
    t,
    adoption_rate,
    where=(t > early_majority_end) & (t <= late_majority_end),
    color="lightcoral",
    alpha=0.6,
    label="Late Majority (34%)",
)
ax2.fill_between(
    t,
    adoption_rate,
    where=(t > late_majority_end),
    color="plum",
    alpha=0.6,
    label="Laggards (16%)",
)

# Add a title and legend
plt.title("Adoption Curve with Adopter Categories")
fig.tight_layout()
# Create a single legend for all plots
handles, labels = [], []
for ax in [ax1, ax2]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
# Remove duplicate labels
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

plt.savefig(os.path.join(SAVE_DIR, "adoption_curve.png"))
plt.close()


print("All plots generated successfully.")
