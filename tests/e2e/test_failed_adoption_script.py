import pytest
from .failed_adoption_script import run_failed_adoption_example

@pytest.mark.e2e
def test_failed_adoption_script():
    df, failed = run_failed_adoption_example()
    # Ensure dataframe has expected shape
    assert df.shape == (50, 3)
    # Product B should be identified as failed (index 1)
    assert 1 in failed
