# Gemini Project Configuration

This file provides project-specific instructions for the Gemini agent.

## Core Principles

*   **Modularity**: The project is structured into distinct modules (`diffuse`, `compete`, `substitute`, etc.). All new code should be placed in the appropriate module.
*   **Pandas Backend**: All pandas DataFrames should be created with the `pyarrow` backend for performance. Use `pd.DataFrame(..., dtype_backend='pyarrow')`.
*   **Computational Efficiency**: Prioritize computationally efficient code. Use vectorized NumPy operations and avoid slow loops. All numerical computations should be routed through the `backend.py` abstraction layer.
*   **API Consistency**: New models should inherit from the appropriate base class (`DiffusionModel` or `MultiProductDiffusionModel`) to maintain a consistent API.
*   **Testing**: All new features must be accompanied by unit tests.

## Rate Limiting
*   To stay within the free tier, do not make excessive API calls. Where possible, use batch-friendly APIs and cache results.

## File Structure

*   Core library code resides in `src/innovate/`.
*   Tests are in the `tests/` directory, mirroring the `src` structure.
*   Examples and tutorials are in the `examples/` directory.
*   High-level documentation (`README.md`, `roadmap.md`, etc.) is in the project root.
