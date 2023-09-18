"""Analyse historical performance
"""

import pandas as pd

from . import constants as const


def perf_summary(
    ret: pd.DataFrame,
    yr=const.YEAR_BY["day"],
    additional_stats={},
    additional_periods={},
):
    stats = {
        "Excess Return": lambda x: x.quant.return_mean(yr),
        "Volatility": lambda x: x.quant.return_vol(yr),
        "Sharpe": lambda x: x.quant.sharpe(yr),
        "Max Drawdown": lambda x: x.quant.max_drawdown()[2],
        **additional_stats,
    }

    idx = ret.index
    periods = {
        "01YR": idx[-yr:],
        "02YR": idx[-2 * yr :],
        "03YR": idx[-3 * yr :],
        "05YR": idx[-5 * yr :],
        "10YR": idx[-10 * yr :],
        "SI": idx,
        **additional_periods,
    }

    data = []
    for s, stat in stats.items():
        for p, period in periods.items():
            for col in ret.columns:
                val = stat(ret[col].loc[period])
                data.append([col, s, p, val])

    summary = pd.DataFrame(
        data, columns=["Asset", "Statistic", "Period", "Value"]
    ).pivot(columns="Statistic", index=["Asset", "Period"], values="Value")

    return summary


def perf_summary_table(ret: pd.DataFrame, yr=const.YEAR_BY["day"]):
    summary = perf_summary(ret, yr)
    table = summary.style.format(
        {
            "Excess Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:,2f}",
            "Max Drawdown": "{:.2%}",
        }
    ).background_gradient()
    return table
