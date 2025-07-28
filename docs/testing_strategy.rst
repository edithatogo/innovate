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

