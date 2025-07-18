import numpy as np
import pandas as pd


def align_index_forward(source, target):
    """Align the source index of a pandas DataFrame or Series to the nearest following timestamps
    in another target pandas DataFrame or Series.

    For each timestamp in the source index, this function finds the first timestamp in the
    target index that is greater than or equal to it, and reindexes the source to these new timestamps.
    If multiple source timestamps map to the same target timestamp, only the last value is kept.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([100, 200], index=pd.to_datetime(['2023-01-02', '2023-01-06']))
        >>> t = pd.Series(range(6), index=pd.to_datetime([
        ...     '2023-01-01',
        ...     '2023-01-03',
        ...     '2023-01-04',
        ...     '2023-01-06',
        ... ]))
        >>> align_index_forward(s, t)
        2023-01-03    100
        2023-01-06    200
        dtype: int64
    """
    if not isinstance(source, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    if not isinstance(target, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a pandas DataFrame or Series")
    if source.index[0] < target.index[0]:
        raise ValueError(
            "Source index start must be greater than or equal to target index start"
        )
    if source.index[-1] > target.index[-1]:
        raise ValueError(
            "Source index end must be less than or equal to target index end"
        )
    source = source.copy()
    insertion_points = target.index.searchsorted(source.index)
    source.index = target.index[insertion_points]  # type: ignore

    # Handle duplicate indices by keeping the last value
    if not source.index.is_unique:
        source = source.groupby(source.index).last()

    return source


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
