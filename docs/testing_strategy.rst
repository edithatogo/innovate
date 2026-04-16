Testing Strategy
===============

This project uses **pytest** for all automated tests. We group tests into
three broad categories:

Unit tests
  Cover the smallest pieces of functionality in isolation. These tests
  should not touch the filesystem or network and should execute very
  quickly.

Integration tests
  Exercise the interaction of several components together. They may read
  from small data files or require more complex setup but still run
  entirely within the Python process.

End-to-end (E2E) tests
  Run through the library as a user would, potentially calling command
  line interfaces or full workflows. These tests tend to be slower and
  may rely on example datasets.

Running tests with coverage
---------------------------

The recommended command to run the full suite with coverage enabled is::

    pytest --cov=innovate --cov-report=term-missing

This reports line coverage for the ``innovate`` package and highlights any
missing lines in the output.

Marking tests
-------------

Please mark tests according to their scope using ``pytest`` markers. Use
``@pytest.mark.unit`` for unit tests, ``@pytest.mark.integration`` for
integration tests and ``@pytest.mark.e2e`` for end-to-end tests. Markers
allow selective running, for example ``pytest -m unit`` runs only unit
tests.

Optional backend tests
----------------------

The project keeps optional backend coverage separate from the base suite.
Use the ``optional_backend`` marker for tests that require JAX, BlackJAX,
ArviZ, or related accelerator dependencies.

Recommended local commands::

    # Base install, no accelerator extras
    uv sync
    pytest -m "not optional_backend" --cov=innovate --cov-report=term-missing

    # Optional accelerator stack
    uv sync --extra jax --extra bayesian
    pytest -m optional_backend --cov=innovate --cov-report=term-missing

The base suite should pass without JAX or Bayesian extras installed. The
optional-backend suite should only be run in environments that explicitly
install the corresponding extras.
