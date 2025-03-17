"""Analyse historical performance"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from . import constants as const
from . import plot


def perf_summary(
    ret: pd.DataFrame,
    bmk=None,
    yr=const.YEAR_BY["day"],
    additional_stats={},
    additional_periods={},
):
    stats = {
        "Excess Return": lambda x, t: x.loc[t].quant.return_mean(yr).iloc[0],
        "Volatility": lambda x, t: x.loc[t].quant.return_vol(yr).iloc[0],
        "Sharpe": lambda x, t: x.loc[t].quant.sharpe(yr).iloc[0],
        "Max Drawdown": lambda x, t: x.loc[t].quant.max_drawdown()[col][2],
        "Hit Ratio": lambda x, t: x.loc[t].quant.hit_ratio().iloc[0],
        **additional_stats,
    }

    if bmk is not None:
        assert isinstance(bmk, pd.DataFrame), "Benchmark must be a DataFrame"
        assert bmk.shape[1] == 1, "Benchmark must have only one column"
        stats["Hit Ratio"] = (
            lambda x, t: x.loc[t].quant.hit_ratio(bmk=bmk.loc[t]).iloc[0]
        )
        stats["Beta"] = lambda x, t: x.loc[t].quant.beta(bmk.loc[t]).iloc[0, 0]
        stats["Alpha"] = lambda x, t: x.loc[t].quant.alpha(bmk.loc[t], yr).iloc[0]
        stats["Tracking Error"] = (
            lambda x, t: x.loc[t].quant.tracking_error(bmk.loc[t], yr).iloc[0]
        )
        stats["Information Ratio"] = (
            lambda x, t: x.loc[t].quant.information_ratio(bmk.loc[t], yr).iloc[0]
        )

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
                val = stat(ret[[col]], period)
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
    table = table.set_caption("Performance Summary")
    return table


def perf_report(
    ret: pd.DataFrame, bmk=None, yr=const.YEAR_BY["day"], window=252, **kwargs
):
    cumret = ret.quant.to_prices()

    print(f"Start date: {str(ret.first_valid_index().date())}")  # type: ignore
    print(f"End date: {str(ret.last_valid_index().date())}")  # type: ignore

    # Summary table
    table = perf_summary_table(ret, bmk=bmk, yr=yr, **kwargs)
    display(table)

    # Drawdowns
    dd = pd.DataFrame(ret.quant.max_drawdown()).T.rename(
        columns={0: "Start Date", 1: "End Date", 2: "Drawdown"}
    )
    dd["Start Date"] = pd.to_datetime(dd["Start Date"]).dt.date
    dd["End Date"] = pd.to_datetime(dd["End Date"]).dt.date
    dd_table = (
        dd.style.format({"Drawdown": "{:.2%}"})
        .background_gradient(cmap="Reds_r", subset=["Drawdown"])
        .set_caption("Drawdowns")
    )
    display(dd_table)

    # Plot cumulative returns
    fig, ax = plt.subplots(figsize=(8, 4))
    cumret.plot(ax=ax, title="Growth of Dollar")
    if bmk is not None:
        bmk.quant.to_prices().plot(ax=ax, color="black", linestyle="--")

    # Plot rolling return
    fig, ax = plt.subplots(figsize=(8, 4))
    ret.rolling(window=window).apply(lambda x: x.quant.return_mean(yr)).iloc[
        window - 1 :
    ].plot(ax=ax, title="Rolling Return")
    plot.set_yaxis_percent(ax)

    # Plot rolling volatility
    fig, ax = plt.subplots(figsize=(8, 4))
    ret.rolling(window=window).apply(lambda x: x.quant.return_vol(yr)).iloc[
        window - 1 :
    ].plot(ax=ax, title="Rolling Volatility")
    plot.set_yaxis_percent(ax)

    # Plot drawdown
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    fig, ax = plt.subplots(figsize=(8, 4))
    drawdown.plot(ax=ax, title="Drawdown")
    ax.hlines(0, drawdown.index[0], drawdown.index[-1], "black", "--")
    plot.set_yaxis_percent(ax)

    # Plot rolling beta to benchmark
    if bmk is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ret.rolling(window=window).apply(lambda x: x.quant.beta(bmk)).iloc[
            window - 1 :
        ].plot(ax=ax, title="Rolling Beta to Benchmark")

    plt.show()

    return None


def rolling_multibeta(
    ret: pd.DataFrame, factors: pd.DataFrame, window=252, statistic: str = "beta"
):
    assert ret.shape[1] == 1, "Returns must have only one column"
    assert factors.index.equals(ret.index), "Index of returns and factors must match"
    assert statistic in ["beta", "corr"], "Invalid statistic"
    beta = pd.DataFrame(index=ret.index, columns=factors.columns)
    for idx in range(window, len(ret)):
        beta.iloc[idx] = ret.iloc[idx + 1 - window : idx + 1].quant.beta(
            factors.iloc[idx + 1 - window : idx + 1]
        )
        if statistic == "corr":
            ret_vol = ret.iloc[idx + 1 - window : idx + 1].quant.return_vol().iloc[0]
            factors_vol = factors.iloc[idx + 1 - window : idx + 1].quant.return_vol()
            beta.iloc[idx] = beta.iloc[idx] / ret_vol * factors_vol
    beta = beta.iloc[window:]
    return beta


# TODO: Add risk decomposition
def risk_decomposition(ret: pd.DataFrame, factors: pd.DataFrame, window=252):
    pass


# TODO: Add risk report
def risk_report():
    pass
