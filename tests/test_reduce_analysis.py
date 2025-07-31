"""
Tests for the reduction analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from innovate.reduce.analysis import identify_reducing_series


class TestReduceAnalysis(unittest.TestCase):
    def test_identify_reducing_series_smoke_test(self):
        """
        A simple smoke test to ensure the main function runs without errors.
        """
        # A series that clearly rises and then falls
        reducing_series = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        # A series that only rises
        rising_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # A flat series
        flat_series = np.ones(11)

        time_series_list = [reducing_series, rising_series, flat_series]

        results_df = identify_reducing_series(time_series_list)

        # Assertions to ensure the output is as expected
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 3)
        self.assertIn("changepoint_index", results_df.columns)
        self.assertIn("trend", results_df.columns)
        self.assertIn("p_value", results_df.columns)

        # Check the results for the clearly reducing series
        self.assertEqual(results_df.loc[0, "trend"], "decreasing")
        self.assertLess(results_df.loc[0, "p_value"], 0.05)  # Expect significance
        self.assertLess(results_df.loc[0, "post_peak_slope"], 0)

        # Check the results for the non-reducing series
        self.assertNotEqual(results_df.loc[1, "trend"], "decreasing")
        self.assertNotEqual(results_df.loc[2, "trend"], "decreasing")


if __name__ == "__main__":
    unittest.main()
