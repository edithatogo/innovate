import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# We mock out pandasdmx at the module level before importing our code,
# because pandasdmx has compatibility issues with newer versions of pydantic.
sys.modules["pandasdmx"] = MagicMock()

from innovate.data.oecd import get_dataset


@patch("innovate.data.oecd.sdmx.Request")
def test_get_dataset_happy_path(mock_request_class):
    # Setup mock
    mock_request_instance = MagicMock()
    mock_data_msg = MagicMock()
    expected_df = pd.DataFrame({"value": [1, 2, 3]})

    mock_data_msg.to_pandas.return_value = expected_df
    mock_request_instance.data.return_value = mock_data_msg
    mock_request_class.return_value = mock_request_instance

    # Call function
    dataset_id = "SOME_DATASET"
    dimensions = {"DIM1": "VAL1"}
    start_date = "2000"
    end_date = "2010"

    result = get_dataset(dataset_id, dimensions, start_date, end_date)

    # Assertions
    mock_request_class.assert_called_once_with("OECD")
    mock_request_instance.data.assert_called_once_with(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    mock_data_msg.to_pandas.assert_called_once()

    pd.testing.assert_frame_equal(result, expected_df)


@patch("innovate.data.oecd.sdmx.Request")
def test_get_dataset_default_dates(mock_request_class):
    # Setup mock
    mock_request_instance = MagicMock()
    mock_data_msg = MagicMock()

    mock_request_instance.data.return_value = mock_data_msg
    mock_request_class.return_value = mock_request_instance

    # Call function with defaults
    dataset_id = "SOME_DATASET"
    dimensions = {"DIM1": "VAL1"}

    get_dataset(dataset_id, dimensions)

    # Assertions
    mock_request_instance.data.assert_called_once_with(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": "1960", "endTime": "2020"},
    )


@patch("innovate.data.oecd.sdmx.Request")
def test_get_dataset_handles_api_error(mock_request_class):
    # Setup mock to raise an exception when requesting data
    mock_request_instance = MagicMock()
    mock_request_instance.data.side_effect = Exception("API Error")
    mock_request_class.return_value = mock_request_instance

    # Call function and expect exception to bubble up
    with pytest.raises(Exception, match="API Error"):
        get_dataset("SOME_DATASET", {})
