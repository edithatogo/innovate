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

## License

This project is licensed under the Apache 2.0 License.
