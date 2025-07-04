# heartflow

A Python library for simplifying innovation/policy diffusion modelling.

This library aims to provide a flexible and robust framework for modeling the diffusion of innovations and policies across various domains, from advanced health technologies (its origin in genetic and genomic testing) to marketing, economics, and policy research.

## Core Features (Initial Release)

*   **Diffusion Models**: Implementations of classic S-curve models like Gompertz, Logistic, and Bass (closed-form solutions).
*   **Generic Competition-Diffusion Framework**: A flexible structure to model multi-product or multi-policy diffusion, allowing for intrinsic adoption rates, market sizes, and interaction matrices (within- and cross-imitation).
*   **Data Handling**: Pandas-friendly API for time-series data with datetime indices.
*   **Estimation**: Initial fitting capabilities using tried-and-true libraries like SciPy for nonlinear least squares.
*   **Seasonality & Dispersion Handling**: Practical strategies for dealing with noisy, seasonal, or over-dispersed raw adoption data, including two-stage decomposition (e.g., using STL) and options for more integrated fitting approaches.

## Roadmap

The `heartflow` library is designed for phased development, with a clear path towards advanced features and XLA-accelerated performance. See the [roadmap.md](roadmap.md) for detailed plans.

## Installation

More information coming soon!

## Usage

Examples and tutorials will be provided to demonstrate how to use the library for various modeling scenarios.
