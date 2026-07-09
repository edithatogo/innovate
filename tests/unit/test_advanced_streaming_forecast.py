import pytest

from innovate.advanced_runtime import update_streaming_forecast


def test_update_streaming_forecast_happy_path():
    """Verify standard streaming forecast update behavior."""
    previous_time = [1.0, 2.0]
    previous_observed = [10.0, 20.0]
    new_time = [3.0, 4.0]
    new_observed = [35.0, 50.0]

    result = update_streaming_forecast(
        previous_time=previous_time,
        previous_observed=previous_observed,
        new_time=new_time,
        new_observed=new_observed,
        assumptions=["stable_growth"],
    )

    payload = result.to_dict()
    assert result.workflow == "streaming_update"
    assert result.stability == "experimental"
    assert payload["time"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["mean"] == [10.0, 20.0, 35.0, 50.0]
    assert payload["metadata"]["previous_count"] == 2
    assert payload["metadata"]["new_count"] == 2
    assert payload["metadata"]["state"]["last_time"] == 4.0
    assert payload["metadata"]["state"]["last_observed"] == 50.0
    assert payload["diagnostics"]["incremental_growth"] == 30.0


def test_update_streaming_forecast_mismatched_previous_lengths():
    """Ensure error is raised when previous time and observed lengths differ."""
    with pytest.raises(ValueError, match="previous_time and previous_observed lengths must match"):
        update_streaming_forecast(
            previous_time=[1.0, 2.0], previous_observed=[10.0], new_time=[3.0], new_observed=[20.0]
        )


def test_update_streaming_forecast_mismatched_new_lengths():
    """Ensure error is raised when new time and observed lengths differ."""
    with pytest.raises(ValueError, match="new_time and new_observed lengths must match"):
        update_streaming_forecast(
            previous_time=[1.0], previous_observed=[10.0], new_time=[2.0, 3.0], new_observed=[20.0]
        )


def test_update_streaming_forecast_unsorted_time():
    """Ensure error is raised when combined time is not sorted."""
    with pytest.raises(ValueError, match="streaming time points must be sorted"):
        update_streaming_forecast(previous_time=[2.0], previous_observed=[10.0], new_time=[1.0], new_observed=[20.0])


def test_update_streaming_forecast_non_cumulative_observed():
    """Ensure error is raised when combined observed values are not cumulative."""
    with pytest.raises(ValueError, match="streaming observed values must be cumulative"):
        update_streaming_forecast(previous_time=[1.0], previous_observed=[20.0], new_time=[2.0], new_observed=[10.0])
