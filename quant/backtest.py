"""Analyse historical performance"""

import numpy as np
import pandas as pd

from . import constants as const


def perf_summary(
    ret: pd.DataFrame,
    bmk=None,
    yr=const.YEAR_BY["day"],
    additional_stats={},
    additional_periods={},
):
    stats = {
        "Excess Return": lambda x: x.quant.return_mean(yr).iloc[0],
        "Volatility": lambda x: x.quant.return_vol(yr).iloc[0],
        "Sharpe": lambda x: x.quant.sharpe(yr).iloc[0],
        "Max Drawdown": lambda x: x.quant.max_drawdown()[col][2],
        "Hit Ratio": lambda x: x.quant.hit_ratio(bmk=bmk).iloc[0],
        **additional_stats,
    }

    if bmk is not None:
        assert isinstance(bmk, pd.DataFrame), "Benchmark must be a DataFrame"
        assert bmk.shape[1] == 1, "Benchmark must have only one column"
        stats["Beta"] = lambda x: x.quant.beta(bmk).iloc[0, 0]
        stats["Alpha"] = lambda x: x.quant.alpha(bmk, yr).iloc[0]
        stats["Tracking Error"] = lambda x: x.quant.tracking_error(bmk, yr).iloc[0]
        stats["Information Ratio"] = lambda x: x.quant.information_ratio(bmk, yr).iloc[
            0
        ]

    idx = ret.index
    periods = {
        "01YR": idx[-yr:],
        "SI": idx,
        **additional_periods,
    }

    num_yr = len(idx) / yr
    if num_yr >= 2:
        periods["02YR"] = idx[-2 * yr :]
        if num_yr >= 3:
            periods["03YR"] = idx[-3 * yr :]
            if num_yr >= 5:
                periods["05YR"] = idx[-5 * yr :]
                if num_yr >= 10:
                    periods["10YR"] = idx[-10 * yr :]

    data = []
    for s, stat in stats.items():
        for p, period in periods.items():
            for col in ret.columns:
                val = stat(ret[[col]].loc[period])
                data.append([col, s, p, val])

    summary = pd.DataFrame(
        data, columns=["Asset", "Statistic", "Period", "Value"]
    ).pivot(columns="Statistic", index=["Asset", "Period"], values="Value")

    return summary


def perf_summary_table(ret: pd.DataFrame, bmk=None, yr=const.YEAR_BY["day"], **kwargs):
    summary = perf_summary(ret, bmk=bmk, yr=yr, **kwargs)
    order = [
        "Excess Return",
        "Volatility",
        "Sharpe",
        "Max Drawdown",
        "Hit Ratio",
    ]
    if bmk is not None:
        order.extend(
            [
                "Beta",
                "Alpha",
                "Tracking Error",
                "Information Ratio",
            ]
        )
    summary = summary.loc[:, np.append(order, np.setdiff1d(summary.columns, order))]
    table = (
        summary.style.format(
            {
                "Excess Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Hit Ratio": "{:.2%}",
                "Beta": "{:.2f}",
                "Alpha": "{:.2%}",
                "Tracking Error": "{:.2%}",
                "Information Ratio": "{:.2f}",
            }
        )
        .background_gradient(subset=["Max Drawdown"], cmap="YlOrRd_r")
        .background_gradient(subset=["Sharpe"], cmap="YlGn")
    )
    if bmk is not None:
        table = table.background_gradient(subset=["Information Ratio"], cmap="YlGn")
    return table
