# Native Conda recipe

This platform-specific recipe replaces the retired pure-Python recipe. It uses
Python 3.15.2, Maturin and Rust and has no pandas/xarray dependency. Channel package
availability, isolated builds and platform consumers are not yet qualified.
Both the build script and publication workflow enforce the migration release gate.
Do not treat the recipe's presence as a validated Conda distribution.
