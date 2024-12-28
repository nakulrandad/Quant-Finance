import numpy as np
import pandas as pd


def fix_index(data):
    """Fix index of the data"""
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    data = data.copy()
    data.index = pd.to_datetime(data.index, format="mixed")
    data.index = data.index.strftime("%Y-%m-%d")
    return data


def fix_low_freq_index(low_freq, high_freq):
    """Move low frequency index to next nearest high frequency index"""
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
