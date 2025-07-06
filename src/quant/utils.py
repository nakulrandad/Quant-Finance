import numpy as np
import pandas as pd


def fix_low_freq_index(low_freq, high_freq):
    """Align low frequency time series index to the nearest following timestamps
    in a higher frequency time series index.

    This function adjusts the index of a lower frequency pandas DataFrame or
    Series so that each timestamp is moved forward to the nearest greater than
    or equal timestamp in the high frequency index.

    Example:
        >>> import pandas as pd
        >>> low = pd.Series([100, 200], index=pd.to_datetime(['2023-01-02', '2023-01-06']))
        >>> high = pd.Series(range(6), index=pd.to_datetime([
        ...     '2023-01-01',
        ...     '2023-01-03',
        ...     '2023-01-04',
        ...     '2023-01-06',
        ... ]))
        >>> fix_low_freq_index(low, high)
        2023-01-03    100
        2023-01-06    200
        dtype: int64
    """
    if not isinstance(low_freq, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    if not isinstance(high_freq, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    assert low_freq.index[0] >= high_freq.index[0]
    assert low_freq.index[-1] <= high_freq.index[-1]
    low_freq = low_freq.copy()
    insertion_points = high_freq.index.searchsorted(low_freq.index)
    low_freq.index = high_freq.index[insertion_points]  # type: ignore
    return low_freq


def prepare_returns(data):
    """Cleanup returns data"""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    data = data.copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    elif isinstance(data, pd.Series):
        data = pd.to_numeric(data, errors="coerce")

    data = data.tz_localize(None)
    return data


def prepare_prices(data, fill_method="ffill"):
    """Cleanup prices data"""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")

    data = data.copy()

    data = data.replace([np.inf, -np.inf], np.nan)

    if fill_method == "ffill":
        data = data.ffill()
    elif fill_method == "bfill":
        data = data.bfill()
    elif fill_method == "zero":
        data = data.fillna(0)
    else:
        raise ValueError(
            "Invalid fill_method. Choose from 'ffill', 'bfill', or 'zero'."
        )

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    elif isinstance(data, pd.Series):
        data = pd.to_numeric(data, errors="coerce")

    data = data.tz_localize(None)
    return data
