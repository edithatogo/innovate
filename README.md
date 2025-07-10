# innovate

A Python library for simplifying innovation and policy diffusion modeling.

This library provides a flexible and robust framework for modeling the complex dynamics of how innovations, technologies, and policies spread over time. It is designed for researchers and practitioners in economics, marketing, public policy, and technology forecasting.

## Core Philosophy

`innovate` is built on a modular architecture, allowing users to combine different models and components to simulate real-world scenarios. The library supports everything from classic S-curve models to advanced agent-based simulations.

## Key Features

*   **Modular Design**: A suite of focused modules for specific modeling tasks:
    *   `innovate.diffuse`: For foundational single-innovation adoption curves (Bass, Gompertz, Logistic).
    *   `innovate.substitute`: For modeling technology replacement and generational products (Fisher-Pry, Norton-Bass).
    *   `innovate.compete`: For analyzing market share dynamics between competing innovations.
    *   `innovate.hype`: For simulating the Gartner Hype Cycle and the impact of public sentiment.
    *   `innovate.fail`: For understanding the mechanisms of failed adoption.
    *   `innovate.adopt`: For classifying adopter types based on their adoption timing.
*   **Efficient Data Handling**: Uses pandas with an Apache Arrow backend for high-performance data manipulation.
*   **Extensible**: Designed with clear base classes to make it easy to add new custom models.
*   **Computationally Aware**: Leverages vectorized NumPy operations for efficiency, with a backend abstraction that will support future acceleration (e.g., with JAX).

## Innovate in the Ecosystem

`innovate` is designed to fill a unique gap in the Python ecosystem. While some libraries offer diffusion models and others provide generic agent-based modeling (ABM) frameworks, `innovate` is the first to integrate both under a unified, domain-specific toolkit for innovation dynamics.

| Feature                       | `innovate`                               | `PyDiM` / `bassmodeldiffusion`      | `Mesa` / `AgentPy` / `BPTK-Py`        |
| ----------------------------- | ---------------------------------------- | ----------------------------------- | ------------------------------------- |
| **Core Diffusion Models**     | ✅ (Bass, Gompertz, Logistic)            | ✅ (Primarily Bass)                 | ❌ (Not its focus)                    |
| **Competition Models**        | ✅ (Lotka-Volterra)                      | ❌                                  | ✅ (Via custom ABM)                   |
| **Substitution Models**       | ✅ (Fisher-Pry)                          | ❌                                  | ✅ (Via custom ABM)                   |
| **Hype Cycle Modeling**       | ✅ (Composite & DDE models)              | ❌                                  | ✅ (Via custom ABM)                   |
| **Agent-Based Modeling**      | ✅ (Integrated with `mesa`)              | ❌                                  | ✅ (Core functionality)               |
| **Pre-configured ABM Scenarios**| ✅ (Competition, Hype, Disruption)       | ❌                                  | ❌ (Requires manual implementation)   |
| **System Dynamics**           | ❌                                       | ❌                                  | ✅ (BPTK-Py only)                     |
| **Unified Framework**         | ✅ (Diffusion + Competition + ABM)       | ❌ (Focused on diffusion)           | ❌ (Focused on ABM/SD)                |
| **Parameter Fitting**         | ✅                                       | ✅                                  | ❌ (Not a primary feature)            |
| **Visualization**             | ✅                                       | ✅                                  | ✅ (Network/Grid plots)               |

This integrated approach means you can start with high-level diffusion models and seamlessly transition to complex, bottom-up agent-based simulations without changing frameworks.

## Roadmap

The `innovate` library is under active development. For detailed plans on upcoming features, including the Agent-Based Modeling (ABM) framework and advanced policy analysis tools, please see our [Roadmap](roadmap.md).

## Installation

```bash
pip install innovate
```
*(Note: The package is not yet available on PyPI under this name, but will be in the future).*

You will also need to install `pyarrow`:
```bash
pip install pyarrow
```

## Usage

Examples and tutorials will be provided in the `examples/` directory to demonstrate how to use the library for various modeling scenarios.

## Example Plots

Here is a sample of the kinds of visualizations you can generate with `innovate`.

| Bass Diffusion | Lotka-Volterra Competition |
| :---: | :---: |
| ![Bass Diffusion Curve](docs/images/bass_diffusion.png) | ![Lotka-Volterra Competition](docs/images/lotka_volterra_competition.png) |

| Hype Cycle | Reduction Analysis |
| :---: | :---: |
| ![Hype Cycle](docs/images/hype_cycle.png) | ![Reduction Analysis](docs/images/reduction_analysis.png) |


## License

This project is licensed under the Apache 2.0 License.