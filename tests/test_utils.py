import pandas as pd
import pytest

from quant import utils


# Fixture for target series
@pytest.fixture
def target_series():
    index = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05", "2023-01-07"])
    return pd.Series(range(4), index=index)


# Test case 1: Basic alignment
def test_align_index_forward_basic(target_series):
    source_index = pd.to_datetime(["2023-01-02", "2023-01-06"])
    source_series = pd.Series([100, 200], index=source_index)
    expected_index = pd.to_datetime(["2023-01-03", "2023-01-07"])
    expected_series = pd.Series([100, 200], index=expected_index)
    result = utils.align_index_forward(source_series, target_series)
    pd.testing.assert_series_equal(result, expected_series)


# Test case 2: Exact match
def test_align_index_forward_exact_match(target_series):
    source_index = pd.to_datetime(["2023-01-01", "2023-01-05"])
    source_series = pd.Series([100, 200], index=source_index)
    result = utils.align_index_forward(source_series, target_series)
    pd.testing.assert_series_equal(result, source_series)


# Test case 3: Multiple source indices mapping to the same target index
def test_align_index_forward_multiple_to_one(target_series):
    source_index = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-06"])
    source_series = pd.Series([100, 200, 300], index=source_index)
    expected_index = pd.to_datetime(["2023-01-03", "2023-01-07"])
    expected_series = pd.Series([200, 300], index=expected_index)
    result = utils.align_index_forward(source_series, target_series)
    pd.testing.assert_series_equal(result, expected_series)
