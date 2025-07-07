import numpy as np
import pandas as pd
import pytest

from quant.quant_accessor import (
    QuantDataFrameAccessor,
    QuantSeriesAccessor,
    shared_manager,
)


# Fixture for sample DataFrame
@pytest.fixture
def sample_dataframe():
    data = {"A": [100, 101, 102, 103, 104], "B": [200, 202, 204, 206, 208]}
    index = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    )
    df = pd.DataFrame(data, index=index)
    return df


# Fixture for sample Series
@pytest.fixture
def sample_series():
    data = [100, 101, 102, 103, 104]
    index = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    )
    series = pd.Series(data, index=index)
    return series


# Test for SharedStateManager and currency methods
def test_set_get_currency():
    initial_currency = shared_manager.get_currency()

    QuantDataFrameAccessor.set_currency("USD")
    assert QuantDataFrameAccessor.get_currency() == "USD"
    assert shared_manager.get_currency() == "USD"

    QuantSeriesAccessor.set_currency("INR")
    assert QuantSeriesAccessor.get_currency() == "INR"
    assert shared_manager.get_currency() == "INR"

    # Restore initial currency
    shared_manager.set_currency(initial_currency)


# Test for Series accessor methods
def test_series_stripna(sample_series):
    series_with_nan = pd.Series([np.nan, 1, 2, 3, np.nan, 4, np.nan], index=range(7))
    expected_series = pd.Series(
        [1, 2, 3, np.nan, 4], index=range(1, 6)
    )  # stripna only removes leading/trailing NaNs
    result = series_with_nan.quant.stripna()
    pd.testing.assert_series_equal(result, expected_series, check_dtype=False)
