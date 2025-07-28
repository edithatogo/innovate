import pytest
from .failed_adoption_script import run_failed_adoption_example

@pytest.mark.e2e
def test_failed_adoption_script():
    """Tests the failed adoption script to ensure it correctly identifies
    a product that fails to meet the adoption threshold.
    """
    df, failed = run_failed_adoption_example()
    # Ensure dataframe has expected shape
    assert df.shape == (50, 3)
    # Product B should be identified as failed (index 1)
    assert failed == [1]
