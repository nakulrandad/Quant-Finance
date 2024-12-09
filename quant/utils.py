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
